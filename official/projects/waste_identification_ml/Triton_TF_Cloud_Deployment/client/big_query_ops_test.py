# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

import unittest
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import big_query_ops


class TestSchemaDefinition(unittest.TestCase):

  def test_schema_definition(self):
    expected_schema = [
        ("particle", "INTEGER", "REQUIRED"),
        ("source_name", "STRING", "REQUIRED"),
        ("image_name", "STRING", "REQUIRED"),
        ("detection_scores", "FLOAT", "REQUIRED"),
        ("creation_time", "STRING", "REQUIRED"),
        ("bbox_0", "INTEGER", "REQUIRED"),
        ("bbox_1", "INTEGER", "REQUIRED"),
        ("bbox_2", "INTEGER", "REQUIRED"),
        ("bbox_3", "INTEGER", "REQUIRED"),
        ("detection_classes", "INTEGER", "REQUIRED"),
        ("detection_classes_names", "STRING", "REQUIRED"),
    ]

    # Check schema length
    self.assertEqual(
        len(big_query_ops._SCHEMA),
        len(expected_schema),
        "Schema length mismatch.",
    )

    # Validate each field's name, type, and mode in order
    for idx, (field, expected) in enumerate(
        zip(big_query_ops._SCHEMA, expected_schema)
    ):
      expected_name, expected_type, expected_mode = expected
      self.assertEqual(
          (field.name, field.field_type, field.mode),
          (expected_name, expected_type, expected_mode),
          f"Mismatch at field index {idx}.",
      )


if __name__ == "__main__":
  unittest.main()
