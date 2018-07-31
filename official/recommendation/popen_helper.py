# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper file for running the async data generation process in OSS."""

import os
import sys


_PYTHON = sys.executable
if not _PYTHON:
  raise RuntimeError("Could not find path to Python interpreter in order to "
                     "spawn subprocesses.")

_ASYNC_GEN_PATH = os.path.join(os.path.dirname(__file__),
                               "data_async_generation.py")

INVOCATION = [_PYTHON, _ASYNC_GEN_PATH]
