# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

from official.projects.waste_identification_ml.Deploy.pet_grading_cloud_deployment import big_query_ops


class BigQueryManagerTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.project_id = "test-project"
    self.dataset_id = "test_dataset"
    self.table_id = "test_table"
    patcher = mock.patch.object(big_query_ops.bigquery, "Client")
    self.mock_bigquery_client_cls = patcher.start()
    self.addCleanup(patcher.stop)
    self.mock_bigquery_client = self.mock_bigquery_client_cls.return_value

  def test_init(self):
    # Act
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )

    # Assert
    self.mock_bigquery_client_cls.assert_called_once_with(
        project=self.project_id
    )
    self.assertEqual(manager.project_id, self.project_id)
    self.assertEqual(manager.dataset_id, self.dataset_id)
    self.assertEqual(manager.table_id, self.table_id)
    self.assertEqual(
        manager.table_ref,
        f"{self.project_id}.{self.dataset_id}.{self.table_id}",
    )

  def test_ensure_dataset_exists(self):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )

    # Act
    manager._ensure_dataset()

    # Assert
    self.mock_bigquery_client.dataset.assert_called_once_with(self.dataset_id)
    self.mock_bigquery_client.get_dataset.assert_called_once_with(
        self.mock_bigquery_client.dataset.return_value
    )
    self.mock_bigquery_client.create_dataset.assert_not_called()

  def test_ensure_dataset_not_found_creates(self):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )
    self.mock_bigquery_client.get_dataset.side_effect = exceptions.NotFound(
        "Dataset not found"
    )

    # Act
    manager._ensure_dataset()

    # Assert
    self.mock_bigquery_client.dataset.assert_called_once_with(self.dataset_id)
    self.mock_bigquery_client.get_dataset.assert_called_once()
    self.mock_bigquery_client.create_dataset.assert_called_once()

  def test_create_table_skips_when_exists_no_overwrite(self):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )

    # Act
    with mock.patch.object(manager, "_ensure_dataset") as mock_ensure_dataset:
      manager.create_table(overwrite=False)
      mock_ensure_dataset.assert_called_once()

    # Assert
    self.mock_bigquery_client.get_table.assert_called_once_with(
        manager.table_ref
    )
    self.mock_bigquery_client.delete_table.assert_not_called()
    self.mock_bigquery_client.create_table.assert_not_called()

  def test_create_table_overwrites_when_exists_and_overwrite(self):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )

    # Act
    with mock.patch.object(manager, "_ensure_dataset") as mock_ensure_dataset:
      manager.create_table(overwrite=True)
      mock_ensure_dataset.assert_called_once()

    # Assert
    self.mock_bigquery_client.get_table.assert_called_once_with(
        manager.table_ref
    )
    self.mock_bigquery_client.delete_table.assert_called_once_with(
        manager.table_ref
    )
    self.mock_bigquery_client.create_table.assert_called_once()

  def test_create_table_creates_when_not_exists(self):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )
    self.mock_bigquery_client.get_table.side_effect = exceptions.NotFound(
        "Table not found"
    )

    # Act
    with mock.patch.object(manager, "_ensure_dataset") as mock_ensure_dataset:
      manager.create_table(overwrite=False)
      mock_ensure_dataset.assert_called_once()

    # Assert
    self.mock_bigquery_client.get_table.assert_called_once_with(
        manager.table_ref
    )
    self.mock_bigquery_client.delete_table.assert_not_called()
    self.mock_bigquery_client.create_table.assert_called_once()

  @mock.patch.object(big_query_ops.pandas_gbq, "to_gbq")
  def test_ingest_data(self, mock_to_gbq):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )
    df = pd.DataFrame([{"tracker_id": 1, "frame_name": "frame_1"}])

    # Act
    manager.ingest_data(df)

    # Assert
    mock_to_gbq.assert_called_once_with(
        df,
        destination_table=manager.table_ref,
        project_id=manager.project_id,
        if_exists="append",
    )

  @mock.patch.object(big_query_ops.subprocess, "run")
  def test_upload_image_results_success(self, mock_run):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )

    # Act
    manager.upload_image_results_to_storage_bucket(
        input_directory="/path/to/my_input_dir",
        prediction_folder="/path/to/my_pred_folder",
        output_directory="gs://my_output_bucket",
    )

    # Assert
    expected_commands = [
        "rm -r my_input_dir",
        "gcloud storage cp -r /path/to/my_pred_folder gs://my_output_bucket",
        "rm -r /path/to/my_pred_folder",
    ]
    mock_run.assert_called_once_with(
        " && ".join(expected_commands), shell=True, check=True
    )

  @mock.patch.object(big_query_ops.subprocess, "run")
  def test_upload_image_results_handles_subprocess_error(self, mock_run):
    # Arrange
    manager = big_query_ops.BigQueryManager(
        self.project_id, self.dataset_id, self.table_id
    )
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="some-command"
    )

    # Act & Assert (Should handle the error gracefully without raising)
    manager.upload_image_results_to_storage_bucket(
        input_directory="/path/to/my_input_dir",
        prediction_folder="/path/to/my_pred_folder",
        output_directory="gs://my_output_bucket",
    )
    mock_run.assert_called_once()


if __name__ == "__main__":
  unittest.main()
