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
import ai.djl.serving.grpc.proto.EmbedRequest;
import ai.djl.serving.grpc.proto.EmbedResponse;

import com.google.protobuf.InvalidProtocolBufferException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Passes generated {@link EmbedRequest} and {@link EmbedResponse} bytes across the worker.
 *
 * <p>The content value is {@code toByteArray} on the way in and {@code parseFrom} on the way out.
 * This class does not define another byte layout.
 */
final class EmbedCodec {

    static final String CONTENT_TYPE = "application/x-protobuf";

    private static final Logger logger = LoggerFactory.getLogger(EmbedCodec.class);

    private static final int ERROR_CODE = 300;

    private EmbedCodec() {}

    static String requestKey() {
        return EmbedRequest.getDescriptor().getFullName();
    }

    static String responseKey() {
        return EmbedResponse.getDescriptor().getFullName();
    }

    /** Writes the request message bytes onto the model input. */
    static void applyRequest(Input input, EmbedRequest request) {
        input.addProperty("Content-Type", CONTENT_TYPE);
        input.add(requestKey(), request.toByteArray());
    }

    static Result decode(Output output) {
        if (output.getCode() >= ERROR_CODE) {
            return Result.response(errorResponse(output));
        }
        byte[] blob = output.getAsBytes(responseKey());
        if (blob == null || blob.length == 0) {
            return failed("EmbedResponse payload is missing");
        }
        EmbedResponse parsed;
        try {
            parsed = EmbedResponse.parseFrom(blob);
        } catch (InvalidProtocolBufferException e) {
            return failed("EmbedResponse payload is invalid");
        }
        if (parsed.getCode() < ERROR_CODE && parsed.getEmbeddingsCount() == 0) {
            return failed("EmbedResponse payload is missing");
        }
        return Result.response(parsed);
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
