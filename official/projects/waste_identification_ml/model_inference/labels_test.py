# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import os
import unittest
from official.projects.waste_identification_ml.model_inference import labels

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")


class LabelsTest(unittest.TestCase):

  def test_read_csv_to_list(self):
    result = labels.read_csv_to_list(f"{TESTDATA}/csv_to_list.csv")

    self.assertEqual(first=result, second=["alpha", "beta", "gamma"])

  def test_categories_dictionary(self):
    expected = {
        1: {"id": 1, "name": "alpha", "supercategory": "objects"},
        2: {"id": 2, "name": "beta", "supercategory": "objects"},
        3: {"id": 3, "name": "gamma", "supercategory": "objects"},
    }

    result = labels.categories_dictionary(["alpha", "beta", "gamma"])

    self.assertEqual(first=result, second=expected)

  def test_load_labels(self):
    label_paths = {
        "1": f"{TESTDATA}/categories_1.csv",
        "2": f"{TESTDATA}/categories_2.csv",
    }
    expected = [
        [
            "Na",
            "alpha",
            "beta",
        ],
        ["Na", "gamma", "delta"],
    ]
    expected_index = {
        1: {"id": 1, "name": "Na_Na", "supercategory": "objects"},
        2: {"id": 2, "name": "Na_delta", "supercategory": "objects"},
        3: {"id": 3, "name": "Na_gamma", "supercategory": "objects"},
        4: {"id": 4, "name": "alpha_Na", "supercategory": "objects"},
        5: {"id": 5, "name": "alpha_delta", "supercategory": "objects"},
        6: {"id": 6, "name": "alpha_gamma", "supercategory": "objects"},
        7: {"id": 7, "name": "beta_Na", "supercategory": "objects"},
        8: {"id": 8, "name": "beta_delta", "supercategory": "objects"},
        9: {"id": 9, "name": "beta_gamma", "supercategory": "objects"},
    }

    result, result_index = labels.load_labels(label_paths)

    self.assertEqual(first=result, second=expected)
    self.assertEqual(first=result_index, second=expected_index)


if __name__ == "__main__":
  unittest.main()
