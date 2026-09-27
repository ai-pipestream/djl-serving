#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import json
import struct
import unittest
import asyncio
from unittest import mock
from unittest.mock import MagicMock, AsyncMock, patch

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.pair_list import PairList


def _make_json_input(payload: dict, properties: dict = None) -> Input:
    inp = Input()
    inp.properties["content-type"] = "application/json"
    if properties:
        inp.properties.update(properties)
    inp.content = PairList()
    inp.content.add(key=None, value=Output._encode_json(payload))
    return inp


def _decode_output(output: Output) -> dict:
    raw = output.content.value_at(0)
    num_pairs = struct.unpack('>h', raw[:2])[0]
    offset = 2
    pairs = {}
    for _ in range(num_pairs):
        key_len = struct.unpack('>i', raw[offset:offset + 4])[0]
        offset += 4
        key = raw[offset:offset + key_len].decode('utf-8')
        offset += key_len
        val_len = struct.unpack('>i', raw[offset:offset + 4])[0]
        offset += 4
        val = raw[offset:offset + val_len].decode('utf-8')
        offset += val_len
        pairs[key] = val
    return pairs


class TestEmbeddingOutputFormatter(unittest.TestCase):

    def setUp(self):
        from djl_python.lmi_vllm.request_response_utils import embedding_output_formatter
        self.formatter = embedding_output_formatter

    def test_single_embedding_from_json_response(self):
        openai_response = {
            "object":
            "list",
            "data": [{
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0
            }],
            "model":
            "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        mock_response = MagicMock()
        mock_response.body = json.dumps(openai_response).encode('utf-8')
        mock_response.status_code = 200

        output = self.formatter(mock_response)
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(data, [[0.1, 0.2, 0.3]])
        self.assertEqual(decoded["last"], "True")

    def test_batch_embeddings_from_json_response(self):
        openai_response = {
            "object":
            "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0
                },
                {
                    "object": "embedding",
                    "embedding": [0.4, 0.5, 0.6],
                    "index": 1
                },
                {
                    "object": "embedding",
                    "embedding": [0.7, 0.8, 0.9],
                    "index": 2
                },
            ],
            "model":
            "test-model",
            "usage": {
                "prompt_tokens": 15,
                "total_tokens": 15
            }
        }
        mock_response = MagicMock()
        mock_response.body = json.dumps(openai_response).encode('utf-8')
        mock_response.status_code = 200

        output = self.formatter(mock_response)
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], [0.1, 0.2, 0.3])
        self.assertEqual(data[1], [0.4, 0.5, 0.6])
        self.assertEqual(data[2], [0.7, 0.8, 0.9])

    def test_high_dimensional_embedding(self):
        embedding = [float(i) / 1000 for i in range(768)]
        openai_response = {"data": [{"embedding": embedding, "index": 0}]}
        mock_response = MagicMock()
        mock_response.body = json.dumps(openai_response).encode('utf-8')
        mock_response.status_code = 200

        output = self.formatter(mock_response)
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(len(data[0]), 768)
        self.assertAlmostEqual(data[0][0], 0.0)
        self.assertAlmostEqual(data[0][767], 0.767)

    def test_error_response_returns_error_output(self):
        error_body = '{"error": "Model not found"}'
        mock_response = MagicMock()
        mock_response.body = error_body.encode('utf-8')
        mock_response.status_code = 404

        output = self.formatter(mock_response)
        decoded = _decode_output(output)
        self.assertIn("error", decoded)
        self.assertEqual(decoded["code"], "404")

    def test_missing_data_field_returns_error_output(self):
        mock_response = MagicMock()
        mock_response.body = json.dumps({
            "error": "something went wrong"
        }).encode('utf-8')
        mock_response.status_code = 200

        output = self.formatter(mock_response)
        decoded = _decode_output(output)
        self.assertIn("error", decoded)
        self.assertEqual(decoded["code"], "500")


class TestTaskToRunnerConvertMapping(unittest.TestCase):

    def setUp(self):
        from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
        self.VllmRbProperties = VllmRbProperties
        self.base_props = {"engine": "Python", "model_id": "some_model"}

    def test_text_embedding_task(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "text-embedding"
            })
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "auto",
            "convert": "embed"
        })

    def test_feature_extraction_task(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "feature-extraction"
            })
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "pooling",
            "convert": "embed"
        })

    def test_generate_task(self):
        props = self.VllmRbProperties(**{
            **self.base_props, "task": "generate"
        })
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "generate",
            "convert": "auto"
        })

    def test_text_generation_maps_to_generate(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "text-generation"
            })
        self.assertEqual(props.task, "generate")
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "generate",
            "convert": "auto"
        })

    def test_auto_task(self):
        props = self.VllmRbProperties(**{**self.base_props, "task": "auto"})
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "auto",
            "convert": "auto"
        })

    def test_classify_task(self):
        props = self.VllmRbProperties(**{
            **self.base_props, "task": "classify"
        })
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "auto",
            "convert": "classify"
        })

    def test_unknown_task_defaults_to_auto(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "something_unknown"
            })
        self.assertEqual(props._map_task_to_runner_convert(), {
            "runner": "auto",
            "convert": "auto"
        })

    def test_default_task_is_auto(self):
        props = self.VllmRbProperties(**self.base_props)
        self.assertEqual(props.task, "auto")


class TestRunnerConvertInEngineArgs(unittest.TestCase):

    def setUp(self):
        from djl_python.properties_manager.vllm_rb_properties import VllmRbProperties
        self.VllmRbProperties = VllmRbProperties
        self.base_props = {"engine": "Python", "model_id": "some_model"}

    def test_text_embedding_task_in_engine_arg_dict(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "text-embedding"
            })
        arg_dict = props.generate_vllm_engine_arg_dict({})
        self.assertEqual(arg_dict["convert"], "embed")
        self.assertEqual(arg_dict["runner"], "auto")

    def test_feature_extraction_in_engine_arg_dict(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "feature-extraction"
            })
        arg_dict = props.generate_vllm_engine_arg_dict({})
        self.assertEqual(arg_dict["convert"], "embed")
        self.assertEqual(arg_dict["runner"], "pooling")

    def test_passthrough_overrides_runner_convert(self):
        props = self.VllmRbProperties(
            **{
                **self.base_props, "task": "text-embedding"
            })
        arg_dict = props.generate_vllm_engine_arg_dict({
            "runner": "pooling",
            "convert": "none"
        })
        self.assertEqual(arg_dict["runner"], "pooling")
        self.assertEqual(arg_dict["convert"], "none")


class TestPreprocessRequestEmbedding(unittest.TestCase):

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_single_text_input(self, mock_extract_lora, mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "test-embed-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        mock_decode.return_value = {"inputs": "hello world"}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": "hello world"})
        result = handler.preprocess_request(inp)

        self.assertIsNotNone(result)
        self.assertEqual(result.vllm_request.input, ["hello world"])
        self.assertEqual(result.vllm_request.model, "test-embed-model")
        self.assertTrue(result.vllm_request.request_id.startswith("embd-"))
        self.assertTrue(result.vllm_request.use_activation)
        self.assertIs(result.inference_invoker, handler.embedding_service)
        self.assertFalse(result.accumulate_chunks)
        self.assertFalse(result.include_prompt)
        self.assertIsNone(result.stream_output_formatter)
        self.assertIsNone(result.lora_request)

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_normalize_false_sets_use_activation_false(self, mock_extract_lora,
                                                       mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = False
        handler.model_name = "test-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        mock_decode.return_value = {"inputs": "test"}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": "test"})
        result = handler.preprocess_request(inp)

        self.assertFalse(result.vllm_request.use_activation)

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_batch_text_input(self, mock_extract_lora, mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "test-embed-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        texts = ["hello", "world", "foo"]
        mock_decode.return_value = {"inputs": texts}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": texts})
        result = handler.preprocess_request(inp)

        self.assertEqual(result.vllm_request.input, texts)
        self.assertEqual(len(result.vllm_request.input), 3)

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_model_name_from_payload(self, mock_extract_lora, mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "default-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        mock_decode.return_value = {"inputs": "test", "model": "custom-model"}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": "test", "model": "custom-model"})
        result = handler.preprocess_request(inp)

        self.assertEqual(result.vllm_request.model, "custom-model")

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_empty_string_input_wrapped_as_list(self, mock_extract_lora,
                                                mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "test-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        mock_decode.return_value = {"inputs": ""}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": ""})
        result = handler.preprocess_request(inp)

        self.assertEqual(result.vllm_request.input, [""])

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_missing_inputs_defaults_to_empty(self, mock_extract_lora,
                                              mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "test-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        mock_decode.return_value = {"model": "test-model"}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"model": "test-model"})
        result = handler.preprocess_request(inp)

        self.assertEqual(result.vllm_request.input, [""])

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_invalid_inputs_type_raises_error(self, mock_extract_lora,
                                              mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "test-model"
        handler.embedding_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None

        mock_decode.return_value = {"inputs": 42}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": 42})
        with self.assertRaises(ValueError):
            handler.preprocess_request(inp)

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_typed_embed_texts_skip_json_decode(self, mock_extract_lora,
                                                mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = True
        handler.model_name = "test-embed-model"
        handler.embedding_service = MagicMock()
        handler.embed_service = MagicMock()
        handler.output_formatter = None
        handler.session_manager = None
        mock_extract_lora.return_value = None

        from djl_python.inference_pb2 import EmbedRequest
        from djl_python.lmi_vllm.request_response_utils import embed_output_formatter
        texts = ["alpha", "bêta"]
        request = EmbedRequest()
        request.inputs.extend(texts)
        inp = Input()
        inp.properties["Content-Type"] = "application/x-protobuf"
        inp.content = PairList()
        inp.content.add(
            key=EmbedRequest.DESCRIPTOR.full_name,
            value=bytearray(request.SerializeToString()))

        result = handler.preprocess_request(inp)

        mock_decode.assert_not_called()
        self.assertEqual(result.vllm_request.input, texts)
        self.assertIs(result.inference_invoker, handler.embed_service)
        self.assertIs(result.non_stream_output_formatter, embed_output_formatter)


class TestEmbeddingInference(unittest.TestCase):

    def _run_async(self, coro):
        return asyncio.run(coro)

    @patch('djl_python.lmi_vllm.vllm_async_service.decode')
    @patch('djl_python.lmi_vllm.vllm_async_service._extract_lora_adapter')
    def test_non_stream_inference_calls_embedding_service(
            self, mock_extract_lora, mock_decode):
        from djl_python.lmi_vllm.vllm_async_service import VLLMHandler
        handler = VLLMHandler()
        handler.is_embedding = True
        handler.normalize_embeddings = False
        handler.model_name = "test-model"
        handler.output_formatter = None
        handler.session_manager = None
        handler.tokenizer = MagicMock()

        openai_body = json.dumps({
            "data": [{
                "embedding": [0.1, 0.2, 0.3],
                "index": 0
            }]
        }).encode('utf-8')
        mock_json_response = MagicMock()
        mock_json_response.body = openai_body
        mock_json_response.status_code = 200

        embedding_service = AsyncMock(return_value=mock_json_response)
        handler.embedding_service = embedding_service

        handler.check_health = AsyncMock()

        mock_decode.return_value = {"inputs": "test sentence"}
        mock_extract_lora.return_value = None

        inp = _make_json_input({"inputs": "test sentence"})
        output = self._run_async(handler.inference(inp))

        embedding_service.assert_called_once()
        call_args = embedding_service.call_args
        request_arg = call_args[0][0]
        self.assertEqual(request_arg.input, ["test sentence"])

        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(data, [[0.1, 0.2, 0.3]])


class TestEmbeddingOutputContract(unittest.TestCase):

    def setUp(self):
        from djl_python.lmi_vllm.request_response_utils import embedding_output_formatter
        self.formatter = embedding_output_formatter

    def _assert_djl_contract(self, output, expected_embeddings):
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), len(expected_embeddings))
        for actual, expected in zip(data, expected_embeddings):
            self.assertIsInstance(actual, list)
            self.assertEqual(actual, expected)

    def test_single_input_returns_list_of_one_embedding(self):
        response = MagicMock()
        response.body = json.dumps({
            "data": [{
                "embedding": [1.0, 2.0, 3.0],
                "index": 0
            }]
        }).encode('utf-8')
        response.status_code = 200

        output = self.formatter(response)
        self._assert_djl_contract(output, [[1.0, 2.0, 3.0]])

    def test_batch_returns_list_matching_batch_size(self):
        embeddings = [[float(i)] * 4 for i in range(8)]
        openai_data = [{
            "embedding": emb,
            "index": i
        } for i, emb in enumerate(embeddings)]
        response = MagicMock()
        response.body = json.dumps({"data": openai_data}).encode('utf-8')
        response.status_code = 200

        output = self.formatter(response)
        self._assert_djl_contract(output, embeddings)

    def test_output_is_json_content_type(self):
        response = MagicMock()
        response.body = json.dumps({
            "data": [{
                "embedding": [1.0],
                "index": 0
            }]
        }).encode('utf-8')
        response.status_code = 200

        output = self.formatter(response)
        self.assertEqual(output.properties.get("Content-Type"),
                         "application/json")


class TestPackedFloat32(unittest.TestCase):

    def test_packed_bytes_match_tensor_buffer(self):
        import torch

        from djl_python.embed_response import (
            embed_response_bytes,
            float32_tensor_bytes,
        )
        from djl_python.inference_pb2 import EmbedResponse
        from djl_python.lmi_vllm.request_response_utils import embed_output_formatter
        from djl_python.lmi_vllm.vllm_async_service import EmbeddingTensors

        # Signaling NaN (0x7f800001) plus 0.25. A Python float round-trip
        # changes the NaN payload to 0x7fc00001.
        raw = bytes.fromhex("0100807f0000803e")
        tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
        self.assertEqual(float32_tensor_bytes(tensor), raw)

        base = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32)
        stepped = base[::2]
        self.assertFalse(stepped.is_contiguous())
        self.assertEqual(
            float32_tensor_bytes(stepped),
            stepped.contiguous().numpy().tobytes())

        wide = torch.tensor([0.25], dtype=torch.float64)
        with self.assertRaises(ValueError):
            float32_tensor_bytes(wide)

        blob = embed_response_bytes(200, [(1, raw)])
        self.assertIn(raw, blob)
        parsed = EmbedResponse()
        parsed.ParseFromString(blob)
        self.assertEqual(parsed.code, 200)
        self.assertEqual(parsed.embeddings[0].index, 1)
        self.assertIn(raw, parsed.embeddings[0].SerializeToString())

        output = embed_output_formatter(EmbeddingTensors([tensor, stepped]))
        message = EmbedResponse()
        key = EmbedResponse.DESCRIPTOR.full_name
        found = None
        for i in range(output.content.size()):
            if output.content.key_at(i) == key:
                found = bytes(output.content.value_at(i))
        self.assertIsNotNone(found)
        message.ParseFromString(found)
        self.assertIn(raw, message.embeddings[0].SerializeToString())
        self.assertIn(
            stepped.contiguous().numpy().tobytes(),
            message.embeddings[1].SerializeToString())
        decoded = _decode_output(output)
        self.assertEqual(decoded["data"], "\n")

        rejected = embed_output_formatter(EmbeddingTensors([wide]))
        self.assertIsNone(self._response_bytes(rejected))
        self.assertEqual(_decode_output(rejected)["code"], "500")

    def _response_bytes(self, output):
        from djl_python.inference_pb2 import EmbedResponse
        key = EmbedResponse.DESCRIPTOR.full_name
        for i in range(output.content.size()):
            if output.content.key_at(i) == key:
                return bytes(output.content.value_at(i))
        return None


if __name__ == '__main__':
    unittest.main()
