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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class EmbedCodecTest {

    @Test
    public void testEncodeRequestTexts() {
        byte[] encoded = EmbedCodec.encodeTexts(List.of("first text", "second \"text\"", "café"));
        ByteBuffer buffer = ByteBuffer.wrap(encoded).order(ByteOrder.LITTLE_ENDIAN);
        Assert.assertEquals(buffer.getInt(), 1);
        Assert.assertEquals(buffer.getInt(), 3);
        Assert.assertEquals(readUtf8(buffer), "first text");
        Assert.assertEquals(readUtf8(buffer), "second \"text\"");
        Assert.assertEquals(readUtf8(buffer), "café");
        Assert.assertFalse(buffer.hasRemaining());

        Input input = new Input();
        EmbedCodec.applyRequest(input, List.of("only"));
        Assert.assertEquals(
                input.getProperty("Content-Type", ""), EmbedCodec.TEXTS_CONTENT_TYPE);
        byte[] payload = input.getAsBytes(EmbedCodec.TEXTS_KEY);
        ByteBuffer one = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        Assert.assertEquals(one.getInt(), 1);
        Assert.assertEquals(one.getInt(), 1);
        Assert.assertEquals(readUtf8(one), "only");
        Assert.assertFalse(one.hasRemaining());
    }

    @Test
    public void testJsonOnlyIsFailedPrecondition() {
        Output output = new Output();
        output.add("[[0.1, 0.2], [0.3, 0.4]]");

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertEquals(result.getFailure(), EmbedCodec.Failure.FAILED_PRECONDITION);
        Assert.assertEquals(result.getDescription(), "embedding-f32 payload is missing");
        Assert.assertNull(result.getResponse());
    }

    @Test
    public void testValidBlob() {
        float[][] vectors = {{0.25f, 0.5f}, {-1.5f, 2.0f}};
        Output output = new Output();
        output.add(EmbedCodec.BINARY_KEY, blob(1, vectors));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        EmbedResponse response = result.getResponse();
        Assert.assertEquals(response.getCode(), 200);
        Assert.assertEquals(response.getEmbeddingsCount(), 2);
        assertVector(response.getEmbeddings(0), 0, 0.25f, 0.5f);
        assertVector(response.getEmbeddings(1), 1, -1.5f, 2.0f);
    }

    @Test
    public void testBlobIgnoresJsonBody() {
        Output output = new Output();
        output.add("[[9.0, 9.0]]");
        output.add(EmbedCodec.BINARY_KEY, blob(1, new float[][] {{0.25f, 0.5f}}));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        assertVector(result.getResponse().getEmbeddings(0), 0, 0.25f, 0.5f);
    }

    @Test
    public void testCorruptBlobIsError() {
        String json = "[[0.25, 0.5]]";

        Output truncated = new Output();
        truncated.add(json);
        truncated.add(EmbedCodec.BINARY_KEY, new byte[] {1, 0, 0, 0});

        Output wrongVersion = new Output();
        wrongVersion.add(json);
        wrongVersion.add(EmbedCodec.BINARY_KEY, blob(2, new float[][] {{0.25f, 0.5f}}));

        ByteBuffer extra = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
        extra.putInt(1);
        extra.putInt(1);
        extra.putInt(1);
        extra.putFloat(9.0f);
        extra.putFloat(8.0f);
        Output lengthMismatch = new Output();
        lengthMismatch.add(json);
        lengthMismatch.add(EmbedCodec.BINARY_KEY, extra.array());

        Output badShape = new Output();
        badShape.add(json);
        ByteBuffer zeroCount = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        zeroCount.putInt(1);
        zeroCount.putInt(0);
        zeroCount.putInt(2);
        badShape.add(EmbedCodec.BINARY_KEY, zeroCount.array());

        Assert.assertEquals(
                EmbedCodec.decode(truncated).getDescription(), "embedding-f32 payload is truncated");
        Assert.assertEquals(
                EmbedCodec.decode(wrongVersion).getDescription(),
                "embedding-f32 version is unsupported");
        Assert.assertEquals(
                EmbedCodec.decode(lengthMismatch).getDescription(),
                "embedding-f32 length does not match count and dimension");
        Assert.assertEquals(
                EmbedCodec.decode(badShape).getDescription(),
                "embedding-f32 count or dimension is invalid");
        for (Output output : List.of(truncated, wrongVersion, lengthMismatch, badShape)) {
            EmbedCodec.Result result = EmbedCodec.decode(output);
            Assert.assertEquals(result.getFailure(), EmbedCodec.Failure.FAILED_PRECONDITION);
            Assert.assertNull(result.getResponse());
        }
    }

    @Test
    public void testErrorOutput() {
        Output output = new Output(503, "overloaded");
        output.add("[[0.1, 0.2]]");
        output.add(EmbedCodec.BINARY_KEY, blob(1, new float[][] {{0.1f, 0.2f}}));

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

    private static String readUtf8(ByteBuffer buffer) {
        int length = buffer.getInt();
        byte[] bytes = new byte[length];
        buffer.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static byte[] blob(int version, float[][] vectors) {
        int count = vectors.length;
        int dimension = vectors[0].length;
        ByteBuffer buffer =
                ByteBuffer.allocate(12 + count * dimension * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(version);
        buffer.putInt(count);
        buffer.putInt(dimension);
        for (float[] vector : vectors) {
            for (float value : vector) {
                buffer.putFloat(value);
            }
        }
        return buffer.array();
    }
}
