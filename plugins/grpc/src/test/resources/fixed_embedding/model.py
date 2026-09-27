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

Embed requests carry length-prefixed UTF-8 texts on ``embed-texts``. The reply
Embed reads is the little-endian ``embedding-f32`` blob. Predict still receives
``[[0.25, 0.5]]`` as JSON. No torch and no GPU.
"""

import array
import json
import struct
import sys

from djl_python import Input
from djl_python import Output

EMBEDDINGS = [[0.25, 0.5]]
TEXTS_KEY = "embed-texts"


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


def _read_texts(blob):
    """Little-endian layout: version 1, text count, then length-prefixed UTF-8."""
    if blob is None or len(blob) < 8:
        raise ValueError("embed-texts payload is truncated")
    version, count = struct.unpack_from("<ii", blob, 0)
    if version != 1:
        raise ValueError("embed-texts version is unsupported")
    if count < 1:
        raise ValueError("embed-texts count is invalid")
    offset = 8
    texts = []
    for _ in range(count):
        if offset + 4 > len(blob):
            raise ValueError("embed-texts payload is truncated")
        (length,) = struct.unpack_from("<i", blob, offset)
        offset += 4
        if length < 0 or offset + length > len(blob):
            raise ValueError("embed-texts payload is truncated")
        texts.append(blob[offset:offset + length].decode("utf-8"))
        offset += length
    if offset != len(blob):
        raise ValueError("embed-texts length does not match count")
    return texts


def handle(inputs: Input):
    """Return JSON for Predict, and the float payload when embed-texts is present."""
    if inputs.is_empty():
        return None

    outputs = Output()
    outputs.add_property("content-type", "application/json")
    outputs.add(json.dumps(EMBEDDINGS) + "\n", key="data")
    if not inputs.contains_key(TEXTS_KEY):
        return outputs
    try:
        texts = _read_texts(inputs.get_as_bytes(key=TEXTS_KEY))
    except (ValueError, UnicodeDecodeError) as exc:
        return Output().error(str(exc))
    if not texts or any(text == "" for text in texts):
        return Output().error("embed-texts is empty")
    outputs.add(_embedding_f32_blob(EMBEDDINGS), key="embedding-f32")
    return outputs
