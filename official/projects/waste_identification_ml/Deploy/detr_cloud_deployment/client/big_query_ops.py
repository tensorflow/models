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

"""Designed to interact with Google BigQuery.

For the purpose of dataset and table management, as well as data ingestion
from pandas DataFrames.
"""

import logging
import os
import subprocess
from google.cloud import bigquery
from google.cloud import exceptions
import pandas as pd
import pandas_gbq

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Centralized Schema Definition
_BIGQUERY_SCHEMA = [
    bigquery.SchemaField("particle", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("source_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("image_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("detection_scores", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("creation_time", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("bbox_0", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_1", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_2", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_3", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("detected_classes", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField(
        "detected_classes_names", "STRING", mode="REQUIRED"
    ),
    bigquery.SchemaField("detected_colors", "STRING", mode="REQUIRED"),
]


class BigQueryManager:
  """Manages interactions with Google BigQuery for dataset and table operations.

  This class provides methods to create datasets and tables, ingest data from
  pandas DataFrames, and manage related file operations in Google Cloud Storage.
  """

  def __init__(self, project_id: str, dataset_id: str, table_id: str):
    """Initializes the BigQuery client and storage coordinates."""
    self.client = bigquery.Client(project=project_id)
    self.project_id = project_id
    self.dataset_id = dataset_id
    self.table_id = table_id
    self.table_ref = f"{project_id}.{dataset_id}.{table_id}"

  def _ensure_dataset(self):
    """Checks if dataset exists, creates it if not."""
    dataset_ref = self.client.dataset(self.dataset_id)
    try:
      self.client.get_dataset(dataset_ref)
    except exceptions.NotFound:
      logging.info("Dataset %s not found. Creating...", self.dataset_id)
      dataset = bigquery.Dataset(dataset_ref)
      self.client.create_dataset(dataset, timeout=30)

  def create_table(self, overwrite: bool = False) -> None:
    """Creates the table with the defined schema."""
    self._ensure_dataset()

    try:
      self.client.get_table(self.table_ref)
      if overwrite:
        logging.info("Overwriting table %s...", self.table_id)
        self.client.delete_table(self.table_ref)
      else:
        logging.info("Table %s already exists. Skipping.", self.table_id)
        return
    except exceptions.NotFound:
      pass

    table = bigquery.Table(self.table_ref, schema=_BIGQUERY_SCHEMA)
    self.client.create_table(table)
    logging.info("Table %s created successfully.", self.table_id)

  def ingest_data(self, df: pd.DataFrame) -> None:
    """Ingests data from a pandas DataFrame into BigQuery using pandas_gbq."""
    pandas_gbq.to_gbq(
        df,
        destination_table=self.table_ref,
        project_id=self.project_id,
        if_exists="append",
    )
    logging.info("Data ingested successfully into %s", self.table_ref)

  def upload_image_results_to_storage_bucket(
      self, input_directory: str, prediction_folder: str, output_directory: str
  ) -> None:
    """Moves folders to the destination bucket and cleans up local directories.

    Args:
        input_directory: Path to the local input directory.
        prediction_folder: Path to the local folder containing results.
        output_directory: The GCS path (gs://...) for output.
    """
    try:
      commands = [
          f"rm -r {os.path.basename(input_directory)}",
          f"gsutil -m cp -r {prediction_folder} {output_directory}",
          f"rm -r {prediction_folder}",
      ]
      subprocess.run(" && ".join(commands), shell=True, check=True)
      logging.info("Successfully moved to destination bucket")
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as e:
      logging.info(
          "Issue in moving folders to destination bucket, due to error : %s", e
      )
