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


class _EmbeddingItem:

    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class _EngineEmbedding:

    def __init__(self, rows, status_code=200):
        self.data = [
            _EmbeddingItem(row, index) for index, row in enumerate(rows)
        ]
        self.status_code = status_code


class _ErrorResponse:

    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body


class TestEmbeddingOutputFormatter(unittest.TestCase):

    def setUp(self):
        from djl_python.lmi_vllm.request_response_utils import embedding_output_formatter
        self.formatter = embedding_output_formatter

    def test_single_embedding_from_engine_response(self):
        output = self.formatter(_EngineEmbedding([[0.1, 0.2, 0.3]]))
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(data, [[0.1, 0.2, 0.3]])
        self.assertEqual(decoded["last"], "True")

    def test_batch_embeddings_from_json_response(self):
        output = self.formatter(
            _EngineEmbedding([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                              [0.7, 0.8, 0.9]]))
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], [0.1, 0.2, 0.3])
        self.assertEqual(data[1], [0.4, 0.5, 0.6])
        self.assertEqual(data[2], [0.7, 0.8, 0.9])

    def test_high_dimensional_embedding(self):
        embedding = [float(i) / 1000 for i in range(768)]
        output = self.formatter(_EngineEmbedding([embedding]))
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(len(data[0]), 768)
        self.assertAlmostEqual(data[0][0], 0.0)
        self.assertAlmostEqual(data[0][767], 0.767)

    def test_error_response_returns_error_output(self):
        output = self.formatter(
            _ErrorResponse(404, '{"error": "Model not found"}'))
        decoded = _decode_output(output)
        self.assertIn("error", decoded)
        self.assertEqual(decoded["code"], "404")

    def test_missing_data_field_returns_error_output(self):
        output = self.formatter(_ErrorResponse(200, "not an embedding object"))
        decoded = _decode_output(output)
        self.assertIn("error", decoded)
        self.assertEqual(decoded["code"], "500")
        self.assertIsNone(self._embed_response(output))

    def _embed_response(self, output):
        from djl_python.inference_pb2 import EmbedResponse
        key = EmbedResponse.DESCRIPTOR.full_name
        for i in range(output.content.size()):
            if output.content.key_at(i) == key:
                message = EmbedResponse()
                message.ParseFromString(bytes(output.content.value_at(i)))
                return message
        return None

    def _assert_embed_response(self, message, embeddings):
        self.assertIsNotNone(message)
        self.assertEqual(message.code, 200)
        self.assertEqual(len(message.embeddings), len(embeddings))
        for item, expected in zip(message.embeddings, embeddings):
            self.assertEqual(len(item.vector), len(expected))
            for actual, value in zip(item.vector, expected):
                self.assertAlmostEqual(actual, float(value), delta=1e-6)

    def test_embed_response_matches_json_vectors(self):
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        output = self.formatter(_EngineEmbedding(embeddings))
        decoded = _decode_output(output)
        self.assertEqual(json.loads(decoded["data"].strip()), embeddings)
        self.assertEqual(decoded["data"], json.dumps(embeddings) + "\n")
        self.assertEqual(decoded["last"], "True")
        self.assertEqual(output.properties.get("Content-Type"), "application/json")
        self._assert_embed_response(self._embed_response(output), embeddings)

    def test_high_dimensional_embed_response_matches_json(self):
        embedding = [float(i) / 1000 for i in range(768)]
        output = self.formatter(_EngineEmbedding([embedding]))
        decoded = _decode_output(output)
        data = json.loads(decoded["data"].strip())
        self.assertEqual(data, [embedding])
        self._assert_embed_response(self._embed_response(output), [embedding])

    def test_error_response_omits_embed_response(self):
        output = self.formatter(_ErrorResponse(404, '{"error": "Model not found"}'))
        decoded = _decode_output(output)
        self.assertEqual(decoded["code"], "404")
        self.assertIsNone(self._embed_response(output))


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
        handler.output_formatter = None
        handler.session_manager = None
        mock_extract_lora.return_value = None

        from djl_python.inference_pb2 import EmbedRequest
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

        embedding_service = AsyncMock(
            return_value=_EngineEmbedding([[0.1, 0.2, 0.3]]))
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
        output = self.formatter(_EngineEmbedding([[1.0, 2.0, 3.0]]))
        self._assert_djl_contract(output, [[1.0, 2.0, 3.0]])

    def test_batch_returns_list_matching_batch_size(self):
        embeddings = [[float(i)] * 4 for i in range(8)]
        output = self.formatter(_EngineEmbedding(embeddings))
        self._assert_djl_contract(output, embeddings)

    def test_output_is_json_content_type(self):
        output = self.formatter(_EngineEmbedding([[1.0]]))
        self.assertEqual(output.properties.get("Content-Type"),
                         "application/json")


if __name__ == '__main__':
    unittest.main()
