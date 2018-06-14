# Copyright 2018 The TensorFlow Global Objectives Authors. All Rights Reserved.
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

"""Runs all unit tests in the Global Objectives package.

Requires that TensorFlow and abseil (https://github.com/abseil/abseil-py) be
installed on your machine. Command to run the tests:
python test_all.py

"""

import os
import sys
import unittest

this_file = os.path.realpath(__file__)
start_dir = os.path.dirname(this_file)
parent_dir = os.path.dirname(start_dir)

sys.path.append(parent_dir)
loader = unittest.TestLoader()
suite = loader.discover(start_dir, pattern='*_test.py')

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
