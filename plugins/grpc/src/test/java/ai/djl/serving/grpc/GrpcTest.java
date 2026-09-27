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
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class GrpcTest {

    private static final Logger logger = LoggerFactory.getLogger(GrpcTest.class);

    @Test
    public void test() throws IOException, ParseException, BadWorkflowException {
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
            "src/test/resources/json_only_embedding"
        };
        CommandLine cmd = parser.parse(options, args, null, false);
        ConfigManager.init(new Arguments(cmd));

        ModelStore store = ModelStore.getInstance();
        store.initialize();

        ModelManager modelManager = ModelManager.getInstance();
        for (Workflow workflow : store.getWorkflows()) {
            modelManager.registerWorkflow(workflow).join();
        }

        GrpcServer server = GrpcServer.newInstance();
        Assert.assertNotNull(server);

        try {
            server.start();
            try (GrpcClient client = GrpcClient.newInstance("localhost:8082")) {
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

                EmbedResponse embed =
                        client.embed("fixed_embedding", List.of("What is Deep Learning?"));
                Assert.assertEquals(embed.getCode(), 200);
                Assert.assertEquals(embed.getEmbeddingsCount(), 1);
                Embedding row = embed.getEmbeddings(0);
                Assert.assertEquals(row.getIndex(), 0);
                Assert.assertEquals(row.getVectorCount(), 2);
                Assert.assertEquals(row.getVector(0), 0.25f, 1.0e-6f);
                Assert.assertEquals(row.getVector(1), 0.5f, 1.0e-6f);

                it =
                        client.inference(
                                "fixed_embedding", "{\"inputs\": [\"What is Deep Learning?\"]}");
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
                contentType =
                        resp.getHeadersOrThrow("content-type").toString(StandardCharsets.UTF_8);
                Assert.assertEquals(contentType, "application/json");

                it =
                        client.inference(
                                "json_only_embedding",
                                "{\"inputs\": [\"What is Deep Learning?\"]}");
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
                Assert.assertEquals(
                        jsonOnly.getStatus().getCode(), Status.Code.FAILED_PRECONDITION);
                Assert.assertEquals(
                        jsonOnly.getStatus().getDescription(), "embedding-f32 payload is missing");

                Iterator<InferenceResponse> ret = client.inference("invalid", "v1", headers, "");
                Assert.assertThrows(ret::next);
            }
        } finally {
            server.stop();
        }
    }
}
