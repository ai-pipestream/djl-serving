/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.grpc;

import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.serving.grpc.proto.EmbedResponse;
import ai.djl.serving.grpc.proto.Embedding;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Optional;

/**
 * Encodes embed requests and decodes worker outputs.
 *
 * <p>The binary content key {@code embedding-f32} is tried first. JSON in {@code Output.getData()}
 * is the fallback, and a disagreement between the two is reported as an internal failure.
 */
final class EmbedCodec {

    static final String BINARY_KEY = "embedding-f32";

    private static final Logger logger = LoggerFactory.getLogger(EmbedCodec.class);

    private static final int VERSION = 1;
    private static final int HEADER_BYTES = 12;
    private static final int ERROR_CODE = 300;
    private static final float VALUE_TOLERANCE = 1.0e-6f;

    private EmbedCodec() {}

    static byte[] encodeRequest(List<String> inputs) {
        JsonArray array = new JsonArray();
        for (String text : inputs) {
            array.add(text);
        }
        JsonObject body = new JsonObject();
        body.add("inputs", array);
        return JsonUtils.GSON_COMPACT.toJson(body).getBytes(StandardCharsets.UTF_8);
    }

    static Result decode(Output output) {
        if (output.getCode() >= ERROR_CODE) {
            return Result.response(errorResponse(output));
        }
        Optional<float[][]> binary = readBinary(output);
        byte[] json = jsonPayload(output);
        if (binary.isPresent()) {
            return finishBinary(output, binary.get(), json);
        }
        return finishJson(output, json);
    }

    private static Optional<float[][]> readBinary(Output output) {
        byte[] blob = output.getAsBytes(BINARY_KEY);
        if (blob == null) {
            return Optional.empty();
        }
        return parseBinary(blob);
    }

    private static Optional<float[][]> parseBinary(byte[] blob) {
        if (blob.length < HEADER_BYTES) {
            logger.warn("embedding-f32 payload is truncated, falling back to JSON");
            return Optional.empty();
        }
        ByteBuffer buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int version = buffer.getInt();
        int count = buffer.getInt();
        int dimension = buffer.getInt();
        if (version != VERSION) {
            logger.warn("embedding-f32 version {} is unsupported, falling back to JSON", version);
            return Optional.empty();
        }
        if (count < 1 || dimension < 1) {
            logger.warn("embedding-f32 count or dimension is invalid, falling back to JSON");
            return Optional.empty();
        }
        long vectorCount = count;
        long vectorDimension = dimension;
        long floats = vectorCount * vectorDimension;
        long expected = HEADER_BYTES + floats * Float.BYTES;
        if (expected != blob.length) {
            logger.warn(
                    "embedding-f32 length does not match count and dimension, falling back to"
                            + " JSON");
            return Optional.empty();
        }
        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = buffer.getFloat();
            }
        }
        return Optional.of(vectors);
    }

    /** A well-formed blob is the result. JSON is parsed only to detect a formatter mismatch. */
    private static Result finishBinary(Output output, float[][] binary, byte[] json) {
        if (json.length == 0) {
            return Result.response(vectorsResponse(output, binary));
        }
        float[][] parsed;
        try {
            parsed = parseJson(json);
        } catch (BadMatrixException e) {
            logger.warn(
                    "embedding-f32 is well formed but JSON is not a matrix: {}", e.getMessage());
            return Result.internal("embedding-f32 does not match JSON embeddings");
        }
        if (!sameVectors(binary, parsed)) {
            return Result.internal("embedding-f32 does not match JSON embeddings");
        }
        return Result.response(vectorsResponse(output, binary));
    }

    private static Result finishJson(Output output, byte[] json) {
        if (json.length == 0) {
            return Result.failedPrecondition("embedding JSON is empty");
        }
        try {
            return Result.response(vectorsResponse(output, parseJson(json)));
        } catch (BadMatrixException e) {
            return Result.failedPrecondition(e.getMessage());
        }
    }

    private static byte[] jsonPayload(Output output) {
        for (Pair<String, BytesSupplier> entry : output.getContent()) {
            if (BINARY_KEY.equals(entry.getKey())) {
                continue;
            }
            BytesSupplier supplier = entry.getValue();
            if (supplier == null) {
                continue;
            }
            byte[] bytes = supplier.getAsBytes();
            if (bytes != null && bytes.length > 0) {
                return bytes;
            }
        }
        return new byte[0];
    }

    private static float[][] parseJson(byte[] json) {
        JsonElement root;
        try {
            root = JsonParser.parseString(new String(json, StandardCharsets.UTF_8));
        } catch (JsonParseException e) {
            throw new BadMatrixException("embedding JSON is not a numeric matrix", e);
        }
        if (root == null || !root.isJsonArray()) {
            throw new BadMatrixException("embedding JSON is not a numeric matrix");
        }
        JsonArray rows = root.getAsJsonArray();
        float[][] vectors = new float[rows.size()][];
        int dimension = -1;
        for (int i = 0; i < rows.size(); i++) {
            JsonElement rowElement = rows.get(i);
            if (rowElement == null || !rowElement.isJsonArray()) {
                throw new BadMatrixException("embedding JSON is not a numeric matrix");
            }
            JsonArray row = rowElement.getAsJsonArray();
            if (dimension < 0) {
                dimension = row.size();
                if (dimension < 1) {
                    throw new BadMatrixException("embedding dimension must be positive");
                }
            } else if (row.size() != dimension) {
                throw new BadMatrixException("embedding vectors have different dimensions");
            }
            float[] vector = new float[dimension];
            for (int j = 0; j < dimension; j++) {
                vector[j] = readNumber(row.get(j));
            }
            vectors[i] = vector;
        }
        return vectors;
    }

    private static float readNumber(JsonElement value) {
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw new BadMatrixException("embedding JSON is not a numeric matrix");
        }
        return value.getAsFloat();
    }

    private static boolean sameVectors(float[][] left, float[][] right) {
        if (left.length != right.length) {
            return false;
        }
        for (int i = 0; i < left.length; i++) {
            if (left[i].length != right[i].length) {
                return false;
            }
            for (int j = 0; j < left[i].length; j++) {
                if (!close(left[i][j], right[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    private static boolean close(float left, float right) {
        if (Float.floatToIntBits(left) == Float.floatToIntBits(right)) {
            return true;
        }
        float delta = left - right;
        return delta <= VALUE_TOLERANCE && delta >= -VALUE_TOLERANCE;
    }

    private static EmbedResponse errorResponse(Output output) {
        EmbedResponse.Builder builder = EmbedResponse.newBuilder().setCode(output.getCode());
        if (output.getMessage() != null) {
            builder.setMessage(output.getMessage());
        }
        return builder.build();
    }

    private static EmbedResponse vectorsResponse(Output output, float[][] vectors) {
        EmbedResponse.Builder builder = EmbedResponse.newBuilder().setCode(output.getCode());
        if (output.getMessage() != null) {
            builder.setMessage(output.getMessage());
        }
        for (int i = 0; i < vectors.length; i++) {
            Embedding.Builder embedding = Embedding.newBuilder().setIndex(i);
            for (float value : vectors[i]) {
                embedding.addVector(value);
            }
            builder.addEmbeddings(embedding);
        }
        return builder.build();
    }

    enum Failure {
        FAILED_PRECONDITION,
        INTERNAL
    }

    static final class Result {

        private final EmbedResponse response;
        private final Failure failure;
        private final String description;

        private Result(EmbedResponse response, Failure failure, String description) {
            this.response = response;
            this.failure = failure;
            this.description = description;
        }

        static Result response(EmbedResponse response) {
            return new Result(response, null, null);
        }

        static Result failedPrecondition(String description) {
            return new Result(null, Failure.FAILED_PRECONDITION, description);
        }

        static Result internal(String description) {
            return new Result(null, Failure.INTERNAL, description);
        }

        EmbedResponse getResponse() {
            return response;
        }

        Failure getFailure() {
            return failure;
        }

        String getDescription() {
            return description;
        }
    }

    private static final class BadMatrixException extends RuntimeException {

        private static final long serialVersionUID = 1L;

        private BadMatrixException(String message) {
            super(message);
        }

        private BadMatrixException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
