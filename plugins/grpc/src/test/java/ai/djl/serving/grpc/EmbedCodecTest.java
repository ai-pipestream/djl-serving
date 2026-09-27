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
import ai.djl.serving.grpc.proto.EmbedResponse;
import ai.djl.serving.grpc.proto.Embedding;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class EmbedCodecTest {

    @Test
    public void testEncodeRequestJson() {
        byte[] encoded = EmbedCodec.encodeRequest(List.of("first text", "second \"text\""));
        JsonElement root = JsonParser.parseString(new String(encoded, StandardCharsets.UTF_8));
        Assert.assertTrue(root.isJsonObject());
        Assert.assertEquals(root.getAsJsonObject().size(), 1);
        JsonArray inputs = root.getAsJsonObject().getAsJsonArray("inputs");
        Assert.assertEquals(inputs.size(), 2);
        Assert.assertEquals(inputs.get(0).getAsString(), "first text");
        Assert.assertEquals(inputs.get(1).getAsString(), "second \"text\"");

        byte[] single = EmbedCodec.encodeRequest(List.of("only"));
        JsonArray one =
                JsonParser.parseString(new String(single, StandardCharsets.UTF_8))
                        .getAsJsonObject()
                        .getAsJsonArray("inputs");
        Assert.assertEquals(one.size(), 1);
        Assert.assertEquals(one.get(0).getAsString(), "only");
    }

    @Test
    public void testJsonMatrix() {
        Output output = new Output();
        output.add("[[0.1, 0.2], [0.3, 0.4]]");

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        EmbedResponse response = result.getResponse();
        Assert.assertEquals(response.getCode(), 200);
        Assert.assertEquals(response.getEmbeddingsCount(), 2);
        assertVector(response.getEmbeddings(0), 0, 0.1f, 0.2f);
        assertVector(response.getEmbeddings(1), 1, 0.3f, 0.4f);
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
    public void testBinaryPreferredWithinTolerance() {
        float binaryValue = 1.0f;
        float jsonValue = binaryValue + 5.0e-7f;
        Assert.assertNotEquals(Float.floatToIntBits(binaryValue), Float.floatToIntBits(jsonValue));
        Assert.assertTrue(Math.abs(binaryValue - jsonValue) <= 1.0e-6f);

        Output output = new Output();
        output.add("[[" + Float.toString(jsonValue) + "]]");
        output.add(EmbedCodec.BINARY_KEY, blob(1, new float[][] {{binaryValue}}));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertNull(result.getFailure());
        Assert.assertEquals(
                Float.floatToIntBits(result.getResponse().getEmbeddings(0).getVector(0)),
                Float.floatToIntBits(binaryValue));
    }

    @Test
    public void testCorruptBlobFallsBack() {
        float[][] jsonVectors = {{0.25f, 0.5f}};
        String json = "[[0.25, 0.5]]";

        Output truncated = new Output();
        truncated.add(json);
        truncated.add(EmbedCodec.BINARY_KEY, new byte[] {1, 0, 0, 0});

        Output wrongVersion = new Output();
        wrongVersion.add(json);
        wrongVersion.add(EmbedCodec.BINARY_KEY, blob(2, new float[][] {{9.0f, 9.0f}}));

        ByteBuffer extra = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
        extra.putInt(1);
        extra.putInt(1);
        extra.putInt(1);
        extra.putFloat(9.0f);
        extra.putFloat(8.0f);
        Output lengthMismatch = new Output();
        lengthMismatch.add(json);
        lengthMismatch.add(EmbedCodec.BINARY_KEY, extra.array());

        for (Output output : List.of(truncated, wrongVersion, lengthMismatch)) {
            EmbedCodec.Result result = EmbedCodec.decode(output);
            Assert.assertNull(result.getFailure());
            EmbedResponse response = result.getResponse();
            Assert.assertEquals(response.getEmbeddingsCount(), 1);
            assertVector(response.getEmbeddings(0), 0, jsonVectors[0][0], jsonVectors[0][1]);
        }
    }

    @Test
    public void testCountMismatchIsInternal() {
        Output output = new Output();
        output.add("[[0.1, 0.2], [0.3, 0.4]]");
        output.add(EmbedCodec.BINARY_KEY, blob(1, new float[][] {{0.1f, 0.2f}}));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertEquals(result.getFailure(), EmbedCodec.Failure.INTERNAL);
        Assert.assertNull(result.getResponse());
        Assert.assertEquals(
                result.getDescription(), "embedding-f32 does not match JSON embeddings");
    }

    @Test
    public void testValueMismatchIsInternal() {
        Output output = new Output();
        output.add("[[0.9, 0.2]]");
        output.add(EmbedCodec.BINARY_KEY, blob(1, new float[][] {{0.1f, 0.2f}}));

        EmbedCodec.Result result = EmbedCodec.decode(output);

        Assert.assertEquals(result.getFailure(), EmbedCodec.Failure.INTERNAL);
        Assert.assertNull(result.getResponse());
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

    @Test
    public void testBadJsonIsFailedPrecondition() {
        Output ragged = new Output();
        ragged.add("[[0.1, 0.2], [0.3]]");
        EmbedCodec.Result raggedResult = EmbedCodec.decode(ragged);
        Assert.assertEquals(raggedResult.getFailure(), EmbedCodec.Failure.FAILED_PRECONDITION);
        Assert.assertEquals(
                raggedResult.getDescription(), "embedding vectors have different dimensions");
        Assert.assertNull(raggedResult.getResponse());

        Output object = new Output();
        object.add("{\"embeddings\": [[0.1, 0.2]]}");
        EmbedCodec.Result objectResult = EmbedCodec.decode(object);
        Assert.assertEquals(objectResult.getFailure(), EmbedCodec.Failure.FAILED_PRECONDITION);
        Assert.assertEquals(
                objectResult.getDescription(), "embedding JSON is not a numeric matrix");
        Assert.assertNull(objectResult.getResponse());
    }

    private static void assertVector(Embedding embedding, int index, float... expected) {
        Assert.assertEquals(embedding.getIndex(), index);
        Assert.assertEquals(embedding.getVectorCount(), expected.length);
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals(embedding.getVector(i), expected[i], 0.0f);
        }
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
