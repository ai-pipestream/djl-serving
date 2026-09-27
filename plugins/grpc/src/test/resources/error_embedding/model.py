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
"""Worker error fixture.

Returns HTTP-style code 503 and no EmbedResponse. No torch and no GPU.
"""

from djl_python import Input
from djl_python import Output


def handle(inputs: Input):
    """Surface a worker error with no vectors."""
    if inputs.is_empty():
        return None
    return Output(code=503, message="overloaded")
