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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.grpc.proto.EmbedResponse;
import ai.djl.serving.grpc.proto.Embedding;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Encodes embed requests and decodes worker outputs.
 *
 * <p>Texts go to the worker as the {@code embed-texts} content key. Vectors come back only from
 * the {@code embedding-f32} content key. This class does not build or parse JSON.
 */
final class EmbedCodec {

    static final String BINARY_KEY = "embedding-f32";
    static final String TEXTS_KEY = "embed-texts";
    static final String TEXTS_CONTENT_TYPE = "application/x-embedding-texts";

    private static final Logger logger = LoggerFactory.getLogger(EmbedCodec.class);

    private static final int VERSION = 1;
    private static final int HEADER_BYTES = 12;
    private static final int TEXT_HEADER_BYTES = 8;
    private static final int ERROR_CODE = 300;

    private EmbedCodec() {}

    /** Writes the repeated texts as a typed content payload. */
    static void applyRequest(Input input, List<String> inputs) {
        input.addProperty("Content-Type", TEXTS_CONTENT_TYPE);
        input.add(TEXTS_KEY, encodeTexts(inputs));
    }

    /**
     * Little-endian layout: version 1, text count, then each text as a byte length and UTF-8
     * bytes.
     */
    static byte[] encodeTexts(List<String> inputs) {
        byte[][] encoded = new byte[inputs.size()][];
        int size = TEXT_HEADER_BYTES;
        for (int i = 0; i < inputs.size(); i++) {
            encoded[i] = inputs.get(i).getBytes(StandardCharsets.UTF_8);
            size += Integer.BYTES + encoded[i].length;
        }
        ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(VERSION);
        buffer.putInt(encoded.length);
        for (byte[] text : encoded) {
            buffer.putInt(text.length);
            buffer.put(text);
        }
        return buffer.array();
    }

    static Result decode(Output output) {
        if (output.getCode() >= ERROR_CODE) {
            return Result.response(errorResponse(output));
        }
        byte[] blob = output.getAsBytes(BINARY_KEY);
        if (blob == null) {
            return failed("embedding-f32 payload is missing");
        }
        return decodeBlob(output, blob);
    }

    private static Result decodeBlob(Output output, byte[] blob) {
        if (blob.length < HEADER_BYTES) {
            return failed("embedding-f32 payload is truncated");
        }
        ByteBuffer buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int version = buffer.getInt();
        int count = buffer.getInt();
        int dimension = buffer.getInt();
        if (version != VERSION) {
            return failed("embedding-f32 version is unsupported");
        }
        if (count < 1 || dimension < 1) {
            return failed("embedding-f32 count or dimension is invalid");
        }
        long vectorCount = count;
        long vectorDimension = dimension;
        long expected = HEADER_BYTES + vectorCount * vectorDimension * Float.BYTES;
        if (expected != blob.length) {
            return failed("embedding-f32 length does not match count and dimension");
        }
        float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = buffer.getFloat();
            }
        }
        return Result.response(vectorsResponse(output, vectors));
    }

    private static Result failed(String description) {
        logger.warn("Rejected embedding payload: {}", description);
        return Result.failedPrecondition(description);
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
        FAILED_PRECONDITION
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
}
