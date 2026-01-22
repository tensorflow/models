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

import subprocess
import unittest
from unittest import mock

from google.cloud import exceptions
import pandas as pd

from official.projects.waste_identification_ml.Deploy.detr_cloud_deployment.client import big_query_ops

MODULE_PATH = big_query_ops.__name__


class BigQueryManagerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_bigquery_client_patch = mock.patch(
        f"{MODULE_PATH}.bigquery.Client"
    )
    self.mock_bigquery_client = self.mock_bigquery_client_patch.start()
    self.mock_pandas_gbq_patch = mock.patch(f"{MODULE_PATH}.pandas_gbq.to_gbq")
    self.mock_pandas_gbq = self.mock_pandas_gbq_patch.start()
    self.mock_subprocess_run_patch = mock.patch(f"{MODULE_PATH}.subprocess.run")
    self.mock_subprocess_run = self.mock_subprocess_run_patch.start()

    self.project_id = "test-project"
    self.dataset_id = "test-dataset"
    self.table_id = "test-table"
    self.manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_init_sets_attributes(self):
    self.assertEqual(self.manager.project_id, self.project_id)
    self.assertEqual(self.manager.dataset_id, self.dataset_id)
    self.assertEqual(self.manager.table_id, self.table_id)
    self.assertEqual(
        self.manager.table_ref,
        f"{self.project_id}.{self.dataset_id}.{self.table_id}",
    )

  def test_init_creates_client(self):
    self.mock_bigquery_client.assert_called_once_with(project=self.project_id)

  def test_ensure_dataset_exists(self):
    self.manager.client.get_dataset.return_value = True

    self.manager._ensure_dataset()

    self.manager.client.get_dataset.assert_called_once_with(
        self.manager.client.dataset(self.dataset_id)
    )
    self.manager.client.create_dataset.assert_not_called()

  @mock.patch(f"{MODULE_PATH}.bigquery.Dataset")
  def test_ensure_dataset_not_found(self, mock_bq_dataset):
    self.manager.client.get_dataset.side_effect = exceptions.NotFound(
        "Dataset not found"
    )
    mock_dataset_ref = self.manager.client.dataset.return_value
    mock_bq_dataset.return_value = "dataset_obj"

    self.manager._ensure_dataset()

    self.manager.client.get_dataset.assert_called_once_with(mock_dataset_ref)
    self.manager.client.dataset.assert_called_once_with(self.dataset_id)
    mock_bq_dataset.assert_called_once_with(mock_dataset_ref)
    self.manager.client.create_dataset.assert_called_once_with(
        "dataset_obj", timeout=30
    )

  @mock.patch(f"{MODULE_PATH}.bigquery.Table")
  def test_create_table_new_table_created_if_not_exists(self, mock_bq_table):
    self.manager.client.get_table.side_effect = exceptions.NotFound(
        "Table not found"
    )
    table_obj = "table_obj"
    mock_bq_table.return_value = table_obj
    with mock.patch.object(self.manager, "_ensure_dataset"):

      self.manager.create_table()

      mock_bq_table.assert_called_once_with(
          self.manager.table_ref, schema=self.manager._schema
      )
      self.manager.client.create_table.assert_called_once_with(table_obj)

  def test_create_table_new_ensures_dataset_exists_if_table_not_exists(self):
    self.manager.client.get_table.side_effect = exceptions.NotFound(
        "Table not found"
    )
    with mock.patch.object(
        self.manager, "_ensure_dataset"
    ) as mock_ensure_dataset:

      self.manager.create_table()

      mock_ensure_dataset.assert_called_once()

  def test_create_table_exists_no_overwrite_does_not_recreate(self):
    self.manager.client.get_table.return_value = True
    with mock.patch.object(self.manager, "_ensure_dataset"):

      self.manager.create_table(overwrite=False)

      self.manager.client.delete_table.assert_not_called()
      self.manager.client.create_table.assert_not_called()

  def test_create_table_exists_no_overwrite_ensures_dataset_exists(self):
    self.manager.client.get_table.return_value = True
    with mock.patch.object(
        self.manager, "_ensure_dataset"
    ) as mock_ensure_dataset:

      self.manager.create_table(overwrite=False)

      mock_ensure_dataset.assert_called_once()

  @mock.patch(f"{MODULE_PATH}.bigquery.Table")
  def test_create_table_exists_overwrite_recreates_table(self, mock_bq_table):
    self.manager.client.get_table.return_value = True
    table_obj = "table_obj"
    mock_bq_table.return_value = table_obj
    with mock.patch.object(self.manager, "_ensure_dataset"):

      self.manager.create_table(overwrite=True)

      self.manager.client.delete_table.assert_called_once_with(
          self.manager.table_ref
      )
      mock_bq_table.assert_called_once_with(
          self.manager.table_ref, schema=self.manager._schema
      )
      self.manager.client.create_table.assert_called_once_with(table_obj)

  def test_create_table_exists_overwrite_ensures_dataset_exists(self):
    self.manager.client.get_table.return_value = True
    with mock.patch.object(
        self.manager, "_ensure_dataset"
    ) as mock_ensure_dataset:

      self.manager.create_table(overwrite=True)

      mock_ensure_dataset.assert_called_once()

  def test_ingest_data(self):
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    self.manager.ingest_data(df)

    self.mock_pandas_gbq.assert_called_once_with(
        df,
        destination_table=self.manager.table_ref,
        project_id=self.project_id,
        if_exists="append",
    )

  def test_upload_image_results_to_storage_bucket_success(self):
    input_dir = "/tmp/input"
    pred_dir = "/tmp/pred"
    output_dir = "gs://bucket/output"

    self.manager.upload_image_results_to_storage_bucket(
        input_dir, pred_dir, output_dir
    )

    self.mock_subprocess_run.assert_called_once()
    args, _ = self.mock_subprocess_run.call_args
    self.assertIn(f"gsutil -m cp -r {pred_dir} {output_dir}", args[0])

  def test_upload_image_results_to_storage_bucket_failure(self):
    input_dir = "/tmp/input"
    pred_dir = "/tmp/pred"
    output_dir = "gs://bucket/output"
    self.mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        1, "cmd"
    )

    with self.assertLogs(level="INFO") as cm:
      self.manager.upload_image_results_to_storage_bucket(
          input_dir, pred_dir, output_dir
      )

    self.mock_subprocess_run.assert_called_once()
    self.assertIn(
        "Issue in moving folders to destination bucket", cm.output[-1]
    )


if __name__ == "__main__":
  unittest.main()
