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
"""Fixed embedding fixture.

Embed requests are serialized EmbedRequest messages. The reply Embed reads is
a serialized EmbedResponse. Each input gets one packed vector, the float32
bytes of 0.25 and 0.5. ``message`` echoes the inputs joined by newlines so a
test can see the protobuf text round-trip. Predict still receives
``[[0.25, 0.5]]`` as JSON. No torch and no GPU.

Every non-empty call appends one line to ``INVOCATIONS``. A request rejected
before ``runJob`` does not touch that file.
"""

from djl_python import Input
from djl_python import Output
from djl_python.embed_response import append_packed_float32
from djl_python.inference_pb2 import EmbedRequest, EmbedResponse

# Little-endian float32 bytes for 0.25, 0.5. This is the tensor buffer.
FLOAT32_BUFFER = bytes.fromhex("0000803e0000003f")
PREDICT_BODY = "[[0.25, 0.5]]\n"
PROTO_CONTENT_TYPE = "application/x-protobuf"
INVOCATIONS = "/tmp/djl-grpc-fixed-embedding-calls"


def handle(inputs: Input):
    """Return JSON for Predict, and EmbedResponse bytes for an EmbedRequest."""
    if inputs.is_empty():
        return None

    with open(INVOCATIONS, "a", encoding="utf-8") as record:
        record.write("1\n")

    outputs = Output()
    outputs.add_property("content-type", "application/json")
    outputs.add(PREDICT_BODY, key="data")
    request_key = EmbedRequest.DESCRIPTOR.full_name
    content_type = inputs.get_property("Content-Type") or ""
    if content_type != PROTO_CONTENT_TYPE and not inputs.contains_key(request_key):
        return outputs
    if not inputs.contains_key(request_key):
        return Output().error("EmbedRequest payload is missing")
    request = EmbedRequest()
    try:
        request.ParseFromString(bytes(inputs.get_as_bytes(key=request_key)))
    except Exception as exc:
        return Output().error(str(exc))
    if not request.inputs or any(text == "" for text in request.inputs):
        return Output().error("EmbedRequest is empty")
    response = EmbedResponse()
    response.code = 200
    response.message = "\n".join(request.inputs)
    for index, _text in enumerate(request.inputs):
        embedding = response.embeddings.add()
        embedding.index = index
        append_packed_float32(embedding, FLOAT32_BUFFER)
    outputs.add(response.SerializeToString(), key=EmbedResponse.DESCRIPTOR.full_name)
    return outputs
