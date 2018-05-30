# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for benchmark_uploader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile
import unittest
from mock import MagicMock
from mock import patch

import tensorflow as tf  # pylint: disable=g-bad-import-order

try:
  from google.cloud import bigquery
  from official.benchmark import benchmark_uploader
except ImportError:
  bigquery = None
  benchmark_uploader = None


@unittest.skipIf(bigquery is None, "Bigquery dependency is not installed.")
class BigQueryUploaderTest(tf.test.TestCase):

  @patch.object(bigquery, "Client")
  def setUp(self, mock_bigquery):
    self.mock_client = mock_bigquery.return_value
    self.mock_dataset = MagicMock(name="dataset")
    self.mock_table = MagicMock(name="table")
    self.mock_client.dataset.return_value = self.mock_dataset
    self.mock_dataset.table.return_value = self.mock_table
    self.mock_client.insert_rows_json.return_value = []

    self.benchmark_uploader = benchmark_uploader.BigQueryUploader()
    self.benchmark_uploader._bq_client = self.mock_client

    self.log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    with open(os.path.join(self.log_dir, "metric.log"), "a") as f:
      json.dump({"name": "accuracy", "value": 1.0}, f)
      f.write("\n")
      json.dump({"name": "loss", "value": 0.5}, f)
      f.write("\n")
    with open(os.path.join(self.log_dir, "run.log"), "w") as f:
      json.dump({"model_name": "value"}, f)

  def tearDown(self):
    tf.gfile.DeleteRecursively(self.get_temp_dir())

  def test_upload_benchmark_run_json(self):
    self.benchmark_uploader.upload_benchmark_run_json(
        "dataset", "table", "run_id", {"model_name": "value"})

    self.mock_client.insert_rows_json.assert_called_once_with(
        self.mock_table, [{"model_name": "value", "model_id": "run_id"}])

  def test_upload_benchmark_metric_json(self):
    metric_json_list = [
        {"name": "accuracy", "value": 1.0},
        {"name": "loss", "value": 0.5}
    ]
    expected_params = [
        {"run_id": "run_id", "name": "accuracy", "value": 1.0},
        {"run_id": "run_id", "name": "loss", "value": 0.5}
    ]
    self.benchmark_uploader.upload_benchmark_metric_json(
        "dataset", "table", "run_id", metric_json_list)
    self.mock_client.insert_rows_json.assert_called_once_with(
        self.mock_table, expected_params)

  def test_upload_benchmark_run_file(self):
    self.benchmark_uploader.upload_benchmark_run_file(
        "dataset", "table", "run_id", os.path.join(self.log_dir, "run.log"))

    self.mock_client.insert_rows_json.assert_called_once_with(
        self.mock_table, [{"model_name": "value", "model_id": "run_id"}])

  def test_upload_metric_file(self):
    self.benchmark_uploader.upload_metric_file(
        "dataset", "table", "run_id",
        os.path.join(self.log_dir, "metric.log"))
    expected_params = [
        {"run_id": "run_id", "name": "accuracy", "value": 1.0},
        {"run_id": "run_id", "name": "loss", "value": 0.5}
    ]
    self.mock_client.insert_rows_json.assert_called_once_with(
        self.mock_table, expected_params)

  def test_insert_run_status(self):
    self.benchmark_uploader.insert_run_status(
        "dataset", "table", "run_id", "status")
    expected_query = ("INSERT dataset.table "
                      "(run_id, status) "
                      "VALUES('run_id', 'status')")
    self.mock_client.query.assert_called_once_with(query=expected_query)

  def test_update_run_status(self):
    self.benchmark_uploader.update_run_status(
        "dataset", "table", "run_id", "status")
    expected_query = ("UPDATE dataset.table "
                      "SET status = 'status' "
                      "WHERE run_id = 'run_id'")
    self.mock_client.query.assert_called_once_with(query=expected_query)


if __name__ == "__main__":
  tf.test.main()
