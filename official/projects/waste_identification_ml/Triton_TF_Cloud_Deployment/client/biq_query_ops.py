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
from google.cloud import bigquery
from google.cloud import exceptions
import pandas as pd
import pandas_gbq

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

_SCHEMA = [
    bigquery.SchemaField("particle", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("source_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("image_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("detection_scores", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("creation_time", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("bbox_0", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_1", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_2", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("bbox_3", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("detection_classes", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("detection_classes_names", "STRING", mode="REQUIRED"),
]


def create_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    overwrite: bool = False,  # New optional argument
) -> None:
  """Creates a table in a BigQuery dataset.

  Args:
      project_id: The Google Cloud project ID.
      dataset_id: The ID of the dataset in which the table is to be created.
      table_id: The ID of the table to be created.
      overwrite: If True, deletes the preexisting table before creating a new
        one.
  """
  client = bigquery.Client(project=project_id)
  dataset_ref = client.dataset(dataset_id)

  try:
    # Check if the dataset already exists
    dataset = client.get_dataset(dataset_ref)
  except exceptions.NotFound:
    # If the dataset does not exist, create it
    dataset = bigquery.Dataset(dataset_ref)
    dataset = client.create_dataset(dataset)

  table_ref = dataset.table(table_id)

  try:
    # Check if the table already exists
    table = client.get_table(table_ref)
    if overwrite:
      logging.info(
          "Overwriting table '%s' in dataset '%s'...", table_id, dataset_id
      )
      client.delete_table(table_ref)
      table = bigquery.Table(table_ref, schema=_SCHEMA)
      client.create_table(table)
      print(f"Table '{table_id}' has been overwritten.")
    else:
      print(f"Table '{table_id}' already exists. Skipping creation.")
  except exceptions.NotFound:
    # If the table does not exist, create it
    table = bigquery.Table(table_ref, schema=_SCHEMA)
    client.create_table(table)
    print(f"Table '{table_id}' created successfully.")


def ingest_data(
    df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str
) -> None:
  """Ingests data from a pandas DataFrame into a specified BigQuery table.

  This function takes a pandas DataFrame and appends its contents to a BigQuery
  table
  identified by the provided dataset and table IDs within the specified project.
  If the table does not exist, BigQuery automatically creates it with a schema
  inferred from the DataFrame.

  Args:
    df: The pandas DataFrame containing the data to be ingested.
    project_id: The Google Cloud project ID.
    dataset_id: The ID of the dataset containing the target table.
    table_id: The ID of the table where the data will be ingested.
  """
  table_ref = f"{project_id}.{dataset_id}.{table_id}"
  pandas_gbq.to_gbq(
      df, destination_table=table_ref, project_id=project_id, if_exists="append"
  )
