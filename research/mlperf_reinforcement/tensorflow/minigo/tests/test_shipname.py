# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import shipname


class TestShipname(unittest.TestCase):
    def test_bootstrap_gen(self):
        name = shipname.generate(0)
        self.assertIn('bootstrap', name)

    def test_detect_name(self):
        string = '000017-model.index'
        detected_name = shipname.detect_model_name(string)
        self.assertEqual(detected_name, '000017-model')

    def test_detect_num(self):
        string = '000017-model.index'
        detected_name = shipname.detect_model_num(string)
        self.assertEqual(detected_name, 17)
