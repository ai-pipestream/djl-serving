#!/usr/bin/env python
#
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
"""Copy a float32 buffer into EmbedResponse's packed vector field."""

from google.protobuf.internal.encoder import TagBytes, _EncodeVarint
from google.protobuf.internal.wire_format import WIRETYPE_LENGTH_DELIMITED

from djl_python.inference_pb2 import EmbedResponse


def append_packed_float32(embedding, payload: bytes) -> None:
    """Copy ``payload`` into ``embedding.vector``.

    The generated setter converts each element through a Python float.
    A packed ``float`` field is the float32 bytes, so those bytes are merged
    into the generated message.
    """
    if len(payload) % 4 != 0:
        raise ValueError("float32 buffer length is not a multiple of 4")
    field = embedding.DESCRIPTOR.fields_by_name["vector"]
    encoded = bytearray(TagBytes(field.number, WIRETYPE_LENGTH_DELIMITED))
    _EncodeVarint(encoded.extend, len(payload))
    encoded.extend(payload)
    embedding.MergeFromString(bytes(encoded))


def embed_response_bytes(code: int, rows) -> bytes:
    """Serialize an EmbedResponse. Each row is ``(index, float32_bytes)``."""
    message = EmbedResponse()
    message.code = code
    for index, payload in rows:
        embedding = message.embeddings.add()
        embedding.index = index
        append_packed_float32(embedding, payload)
    return message.SerializeToString()


def float32_tensor_bytes(tensor) -> bytes:
    """Return the tensor's float32 buffer.

    A device-to-host copy is performed when the tensor is not already on CPU.
    The dtype is not changed.
    """
    import numpy as np
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ValueError("embedding output is not a tensor")
    if tensor.dtype != torch.float32:
        raise ValueError("embedding tensor dtype is %s" % tensor.dtype)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("embedding output is not a float vector")
    if tensor.device.type != "cpu":
        tensor = tensor.detach().to(device="cpu")
    else:
        tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    array = tensor.numpy()
    if array.dtype != np.float32 or array.dtype.str != "<f4":
        raise ValueError("embedding buffer is not little-endian float32")
    return array.tobytes()
