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
import ai.djl.serving.grpc.proto.Embedding;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.List;

public class EmbedCodecTest {

    @Test
    public void testEncodeRequestTexts() throws Exception {
        EmbedRequest request =
                EmbedRequest.newBuilder()
                        .setModelName("fixed_embedding")
                        .addInputs("first text")
                        .addInputs("second \"text\"")
                        .addInputs("café")
                        .build();
        Input input = new Input();
        EmbedCodec.applyRequest(input, request);

        Assert.assertEquals(input.getProperty("Content-Type", ""), EmbedCodec.CONTENT_TYPE);
        Assert.assertEquals(
                EmbedCodec.requestKey(), "ai.djl.serving.grpc.proto.EmbedRequest");
        Assert.assertEquals(
                EmbedCodec.responseKey(), "ai.djl.serving.grpc.proto.EmbedResponse");
        EmbedRequest parsed = EmbedRequest.parseFrom(input.getAsBytes(EmbedCodec.requestKey()));
        Assert.assertEquals(parsed.getModelName(), "fixed_embedding");
        Assert.assertEquals(parsed.getInputsList(), List.of("first text", "second \"text\"", "café"));
    }

    @Test
    public void testJsonOnlyIsFailedPrecondition() {
        Output output = new Output();
        output.add("[[0.1, 0.2], [0.3, 0.4]]");

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertEquals(result.getFailure(), EmbedCodec.Failure.FAILED_PRECONDITION);
        Assert.assertEquals(result.getDescription(), "EmbedResponse payload is missing");
        Assert.assertNull(result.getResponse());
    }

    @Test
    public void testValidBlob() {
        Output output = new Output();
        output.add(EmbedCodec.responseKey(), responseBytes(0.25f, 0.5f));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        EmbedResponse response = result.getResponse();
        Assert.assertEquals(response.getCode(), 200);
        Assert.assertEquals(response.getEmbeddingsCount(), 1);
        assertVector(response.getEmbeddings(0), 0, 0.25f, 0.5f);
    }

    @Test
    public void testBlobIgnoresJsonBody() {
        Output output = new Output();
        output.add("[[9.0, 9.0]]");
        output.add(EmbedCodec.responseKey(), responseBytes(0.25f, 0.5f));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        assertVector(result.getResponse().getEmbeddings(0), 0, 0.25f, 0.5f);
    }

    @Test
    public void testCorruptBlobIsError() {
        Output truncated = new Output();
        truncated.add("[[0.25, 0.5]]");
        // Field 3, length-delimited, claims 10 bytes that are not present.
        truncated.add(EmbedCodec.responseKey(), new byte[] {26, 10});

        Output empty = new Output();
        empty.add(EmbedCodec.responseKey(), EmbedResponse.getDefaultInstance().toByteArray());

        Assert.assertEquals(
                EmbedCodec.decode(truncated).getDescription(), "EmbedResponse payload is invalid");
        Assert.assertEquals(
                EmbedCodec.decode(empty).getDescription(), "EmbedResponse payload is missing");
        for (Output output : List.of(truncated, empty)) {
            EmbedCodec.Result result = EmbedCodec.decode(output);
            Assert.assertEquals(result.getFailure(), EmbedCodec.Failure.FAILED_PRECONDITION);
            Assert.assertNull(result.getResponse());
        }
    }

    @Test
    public void testErrorOutput() {
        Output output = new Output(503, "overloaded");
        output.add("[[0.1, 0.2]]");
        output.add(EmbedCodec.responseKey(), responseBytes(0.1f, 0.2f));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        EmbedResponse response = result.getResponse();
        Assert.assertEquals(response.getCode(), 503);
        Assert.assertEquals(response.getMessage(), "overloaded");
        Assert.assertEquals(response.getEmbeddingsCount(), 0);
    }

    private static void assertVector(Embedding embedding, int index, float... expected) {
        Assert.assertEquals(embedding.getIndex(), index);
        Assert.assertEquals(embedding.getVectorCount(), expected.length);
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(embedding.getVector(i), expected[i], 0.0f);
        }
    }

    private static byte[] responseBytes(float... vector) {
        Embedding.Builder embedding = Embedding.newBuilder().setIndex(0);
        for (float value : vector) {
            embedding.addVector(value);
        }
        return EmbedResponse.newBuilder().setCode(200).addEmbeddings(embedding).build().toByteArray();
    }
}
