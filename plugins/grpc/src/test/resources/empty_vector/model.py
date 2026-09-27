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
"""Empty-vector fixture.

Returns an EmbedResponse whose embedding has a zero-length packed vector.
No torch and no GPU.
"""

from djl_python import Input
from djl_python import Output
from djl_python.inference_pb2 import EmbedRequest, EmbedResponse

PROTO_CONTENT_TYPE = "application/x-protobuf"


def handle(inputs: Input):
    """Return one embedding and no float32 bytes."""
    if inputs.is_empty():
        return None
    request_key = EmbedRequest.DESCRIPTOR.full_name
    content_type = inputs.get_property("Content-Type") or ""
    if content_type != PROTO_CONTENT_TYPE and not inputs.contains_key(request_key):
        outputs = Output()
        outputs.add_property("content-type", "application/json")
        outputs.add("[]\n", key="data")
        return outputs
    response = EmbedResponse()
    response.code = 200
    embedding = response.embeddings.add()
    embedding.index = 0
    outputs = Output()
    outputs.add(response.SerializeToString(), key=EmbedResponse.DESCRIPTOR.full_name)
    return outputs
