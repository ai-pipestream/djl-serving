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

Returns ``[[0.25, 0.5]]`` as JSON ``data`` and the little-endian ``embedding-f32``
blob the vLLM formatter attaches. No torch and no GPU.
"""

import array
import json
import struct
import sys

from djl_python import Input
from djl_python import Output

EMBEDDINGS = [[0.25, 0.5]]


def _embedding_f32_blob(embeddings):
    """Little-endian layout: version 1, vector count, dimension, row-major float32."""
    count = len(embeddings)
    dimension = len(embeddings[0])
    flat = array.array("f")
    for row in embeddings:
        flat.extend(float(value) for value in row)
    if sys.byteorder != "little":
        flat.byteswap()
    return struct.pack("<iii", 1, count, dimension) + flat.tobytes()


def handle(inputs: Input):
    """Return the fixed matrix on the normal Output path."""
    if inputs.is_empty():
        return None

    # Same JSON text the formatter puts in the async envelope's data field.
    body = json.dumps(EMBEDDINGS) + "\n"
    outputs = Output()
    outputs.add_property("content-type", "application/json")
    outputs.add(body, key="data")
    outputs.add(_embedding_f32_blob(EMBEDDINGS), key="embedding-f32")
    return outputs
