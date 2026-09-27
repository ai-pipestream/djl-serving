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
"""Invalid EmbedResponse fixture.

Writes a truncated protobuf blob under the EmbedResponse content key.
No torch and no GPU.
"""

from djl_python import Input
from djl_python import Output
from djl_python.inference_pb2 import EmbedResponse

# Field 3, length-delimited, claims 10 bytes that are not present.
TRUNCATED = bytes((26, 10))


def handle(inputs: Input):
    """Return bytes that are not a valid EmbedResponse."""
    if inputs.is_empty():
        return None
    outputs = Output()
    outputs.add(TRUNCATED, key=EmbedResponse.DESCRIPTOR.full_name)
    return outputs
