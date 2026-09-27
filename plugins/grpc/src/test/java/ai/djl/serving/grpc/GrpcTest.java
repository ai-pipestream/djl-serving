/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.serving.Arguments;
import ai.djl.serving.GrpcServer;
import ai.djl.serving.grpc.proto.EmbedResponse;
import ai.djl.serving.grpc.proto.Embedding;
import ai.djl.serving.grpc.proto.InferenceResponse;
import ai.djl.serving.grpc.proto.PingResponse;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.ModelStore;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class GrpcTest {

    private static final Logger logger = LoggerFactory.getLogger(GrpcTest.class);

    /** Little-endian float32 bytes for 0.25, 0.5. The fixture copies this buffer. */
    private static final byte[] FLOAT32_025_05 = {
        0x00, 0x00, (byte) 0x80, 0x3e, 0x00, 0x00, 0x00, 0x3f
    };

    private static final Path INVOCATIONS = Path.of("/tmp/djl-grpc-fixed-embedding-calls");

    private GrpcServer server;
    private GrpcClient client;

    @BeforeClass
    public void start() throws IOException, ParseException, BadWorkflowException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();
        String[] args = {
            "-m",
            "../../engines/python/src/test/resources/rolling_batch",
            "-m",
            "../../engines/python/src/test/resources/echo",
            "-m",
            "src/test/resources/fixed_embedding",
            "-m",
            "src/test/resources/json_only_embedding",
            "-m",
            "src/test/resources/error_embedding",
            "-m",
            "src/test/resources/invalid_embed",
            "-m",
            "src/test/resources/empty_vector"
        };
        CommandLine cmd = parser.parse(options, args, null, false);
        ConfigManager.init(new Arguments(cmd));

        ModelStore store = ModelStore.getInstance();
        store.initialize();

        ModelManager modelManager = ModelManager.getInstance();
        for (Workflow workflow : store.getWorkflows()) {
            modelManager.registerWorkflow(workflow).join();
        }

        server = GrpcServer.newInstance();
        Assert.assertNotNull(server);
        server.start();
        client = GrpcClient.newInstance("localhost:8082");
    }

    @AfterClass
    public void stop() throws IOException {
        if (client != null) {
            client.close();
        }
        if (server != null) {
            server.stop();
        }
    }

    @Test
    public void test() {
        PingResponse ping = client.ping();
        logger.info("Ping response: {}", ping);
        Assert.assertEquals(ping.getCode(), 200);

        String data = "{\"inputs\": \"request_0\", \"parameters\": {\"max_length\": 5}}";
        Iterator<InferenceResponse> it = client.inference("rolling_batch", data);
        InferenceResponse resp = it.next();
        logger.info("inference response: {}", resp);
        Assert.assertEquals(resp.getCode(), 200);
        while (it.hasNext()) {
            resp = it.next();
            logger.info("inference response: {}", resp);
        }

        Map<String, String> headers = Map.of("content-type", "application/json");
        it = client.inference("echo", null, headers, "hello");
        resp = it.next();
        Assert.assertEquals(resp.getCode(), 200);
        Assert.assertFalse(it.hasNext());
        Assert.assertEquals(resp.getHeadersCount(), 1);
        Assert.assertEquals(resp.getOutput().toString(StandardCharsets.UTF_8), "hello");
        String contentType =
                resp.getHeadersOrThrow("content-type").toString(StandardCharsets.UTF_8);
        Assert.assertEquals(contentType, "application/json");

        EmbedResponse embed = client.embed("fixed_embedding", List.of("What is Deep Learning?"));
        Assert.assertEquals(embed.getCode(), 200);
        Assert.assertEquals(embed.getEmbeddingsCount(), 1);
        assertExactFloat32(embed.getEmbeddings(0), 0);

        it = client.inference("fixed_embedding", "{\"inputs\": [\"What is Deep Learning?\"]}");
        resp = it.next();
        Assert.assertEquals(resp.getCode(), 200);
        Assert.assertFalse(it.hasNext());
        String predictBody = resp.getOutput().toString(StandardCharsets.UTF_8);
        JsonElement parsed = JsonParser.parseString(predictBody);
        Assert.assertTrue(parsed.isJsonArray());
        JsonArray matrix = parsed.getAsJsonArray();
        Assert.assertEquals(matrix.size(), 1);
        JsonArray vector = matrix.get(0).getAsJsonArray();
        Assert.assertEquals(vector.size(), 2);
        Assert.assertEquals(vector.get(0).getAsFloat(), 0.25f, 1.0e-6f);
        Assert.assertEquals(vector.get(1).getAsFloat(), 0.5f, 1.0e-6f);
        contentType = resp.getHeadersOrThrow("content-type").toString(StandardCharsets.UTF_8);
        Assert.assertEquals(contentType, "application/json");

        it = client.inference("json_only_embedding", "{\"inputs\": [\"What is Deep Learning?\"]}");
        resp = it.next();
        Assert.assertEquals(resp.getCode(), 200);
        Assert.assertFalse(it.hasNext());
        parsed = JsonParser.parseString(resp.getOutput().toString(StandardCharsets.UTF_8));
        Assert.assertTrue(parsed.isJsonArray());
        StatusRuntimeException jsonOnly =
                Assert.expectThrows(
                        StatusRuntimeException.class,
                        () ->
                                client.embed(
                                        "json_only_embedding",
                                        List.of("What is Deep Learning?")));
        Assert.assertEquals(jsonOnly.getStatus().getCode(), Status.Code.FAILED_PRECONDITION);
        Assert.assertEquals(jsonOnly.getStatus().getDescription(), "EmbedResponse payload is missing");

        Iterator<InferenceResponse> ret = client.inference("invalid", "v1", headers, "");
        Assert.assertThrows(ret::next);
    }

    @Test
    public void testEmptyInputsAreInvalidArgument() throws IOException {
        long before = invocationBytes();
        StatusRuntimeException error =
                Assert.expectThrows(
                        StatusRuntimeException.class,
                        () -> client.embed("fixed_embedding", List.of()));
        Assert.assertEquals(error.getStatus().getCode(), Status.Code.INVALID_ARGUMENT);
        Assert.assertEquals(error.getStatus().getDescription(), "inputs must be non-empty strings");
        Assert.assertEquals(invocationBytes(), before);
    }

    @Test
    public void testBlankInputIsInvalidArgument() throws IOException {
        long before = invocationBytes();
        for (String blank : List.of("", " ", "\t\n")) {
            StatusRuntimeException error =
                    Assert.expectThrows(
                            StatusRuntimeException.class,
                            () -> client.embed("fixed_embedding", List.of(blank)));
            Assert.assertEquals(error.getStatus().getCode(), Status.Code.INVALID_ARGUMENT);
            Assert.assertEquals(
                    error.getStatus().getDescription(), "inputs must be non-empty strings");
        }
        Assert.assertEquals(invocationBytes(), before);
    }

    @Test
    public void testUnknownModelIsNotFound() {
        StatusRuntimeException error =
                Assert.expectThrows(
                        StatusRuntimeException.class,
                        () -> client.embed("no-such-model", List.of("hello")));
        Assert.assertEquals(error.getStatus().getCode(), Status.Code.NOT_FOUND);
        Assert.assertEquals(
                error.getStatus().getDescription(), "Model or workflow not found: no-such-model");
    }

    @Test
    public void testOneText() {
        EmbedResponse embed = client.embed("fixed_embedding", List.of("one"));
        Assert.assertEquals(embed.getCode(), 200);
        Assert.assertEquals(embed.getMessage(), "one");
        Assert.assertEquals(embed.getEmbeddingsCount(), 1);
        assertExactFloat32(embed.getEmbeddings(0), 0);
    }

    @Test
    public void testSeveralTexts() {
        List<String> texts = List.of("alpha", "beta", "gamma");
        EmbedResponse embed = client.embed("fixed_embedding", texts);
        Assert.assertEquals(embed.getCode(), 200);
        Assert.assertEquals(embed.getMessage(), "alpha\nbeta\ngamma");
        Assert.assertEquals(embed.getEmbeddingsCount(), texts.size());
        for (int i = 0; i < texts.size(); i++) {
            assertExactFloat32(embed.getEmbeddings(i), i);
        }
    }

    @Test
    public void testNonAsciiTextRoundTrips() {
        List<String> texts = List.of("café", "東京", "naïve");
        EmbedResponse embed = client.embed("fixed_embedding", texts);
        Assert.assertEquals(embed.getCode(), 200);
        Assert.assertEquals(embed.getMessage(), "café\n東京\nnaïve");
        Assert.assertEquals(embed.getEmbeddingsCount(), texts.size());
        for (int i = 0; i < texts.size(); i++) {
            assertExactFloat32(embed.getEmbeddings(i), i);
        }
    }

    @Test
    public void testInvalidProtobufIsFailedPrecondition() {
        StatusRuntimeException error =
                Assert.expectThrows(
                        StatusRuntimeException.class,
                        () -> client.embed("invalid_embed", List.of("hello")));
        Assert.assertEquals(error.getStatus().getCode(), Status.Code.FAILED_PRECONDITION);
        Assert.assertEquals(error.getStatus().getDescription(), "EmbedResponse payload is invalid");
    }

    @Test
    public void testWorkerErrorOmitsVectors() {
        EmbedResponse embed = client.embed("error_embedding", List.of("hello"));
        Assert.assertEquals(embed.getCode(), 503);
        Assert.assertEquals(embed.getMessage(), "overloaded");
        Assert.assertEquals(embed.getEmbeddingsCount(), 0);
    }

    @Test
    public void testEmptyVectorIsRejected() {
        StatusRuntimeException error =
                Assert.expectThrows(
                        StatusRuntimeException.class,
                        () -> client.embed("empty_vector", List.of("hello")));
        Assert.assertEquals(error.getStatus().getCode(), Status.Code.FAILED_PRECONDITION);
        Assert.assertEquals(error.getStatus().getDescription(), "EmbedResponse vector is empty");
    }

    @Test
    public void testExactFloat32Bits() {
        EmbedResponse embed = client.embed("fixed_embedding", List.of("bits"));
        assertExactFloat32(embed.getEmbeddings(0), 0);
    }

    private static long invocationBytes() throws IOException {
        if (!Files.exists(INVOCATIONS)) {
            return 0;
        }
        return Files.size(INVOCATIONS);
    }

    /**
     * The packed field is the trailing float32 bytes of the serialized embedding. Index 0 is
     * omitted by proto3, so a one-element message is tag, length, then those bytes.
     */
    private static void assertExactFloat32(Embedding row, int index) {
        Assert.assertEquals(row.getIndex(), index);
        Assert.assertEquals(row.getVectorCount(), FLOAT32_025_05.length / 4);
        byte[] serialized = row.toByteArray();
        int payloadStart = serialized.length - FLOAT32_025_05.length;
        Assert.assertTrue(payloadStart >= 2);
        Assert.assertEquals(serialized[payloadStart - 2], (byte) 0x12);
        Assert.assertEquals(serialized[payloadStart - 1], (byte) FLOAT32_025_05.length);
        Assert.assertEquals(
                Arrays.copyOfRange(serialized, payloadStart, serialized.length), FLOAT32_025_05);
    }
}
