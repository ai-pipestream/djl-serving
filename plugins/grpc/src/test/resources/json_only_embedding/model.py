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
"""JSON-only embedding fixture.

Predict receives a JSON matrix. There is no embedding-f32 payload, so Embed
must not treat this output as success. No torch and no GPU.
"""

import json

from djl_python import Input
from djl_python import Output

EMBEDDINGS = [[0.25, 0.5]]


def handle(inputs: Input):
    """Return the JSON matrix and nothing else."""
    if inputs.is_empty():
        return None

    outputs = Output()
    outputs.add_property("content-type", "application/json")
    outputs.add(json.dumps(EMBEDDINGS) + "\n", key="data")
    return outputs
