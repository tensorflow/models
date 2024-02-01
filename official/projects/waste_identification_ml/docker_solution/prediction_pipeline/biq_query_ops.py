# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from google.cloud import bigquery
from google.cloud import exceptions
from google.cloud.bigquery import schema
import pandas as pd


def create_table(
    table_schema: list[schema.SchemaField],
    project_id: str,
    dataset_id: str,
    table_id: str
) -> None:
  """Creates a table in a BigQuery dataset.

  This function checks if the specified dataset exists within the given
  project. If not, it creates the dataset. Then, it checks if the specified
  table exists within the dataset. If not, it creates the table using the
  provided schema.

  Args:
    table_schema: A list of SchemaField objects representing the schema of the
    table.
    project_id: The Google Cloud project ID.
    dataset_id: The ID of the dataset in which the table is to be created.
    table_id: The ID of the table to be created.
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
    client.get_table(table_ref)
  except exceptions.NotFound:
    # If the table does not exist, create it
    table = bigquery.Table(table_ref, schema=table_schema)
    client.create_table(table)


def ingest_data(
    df: pd.DataFrame,
    project_id: str,
    dataset_id: str,
    table_id: str
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
  df.to_gbq(
      destination_table=table_ref, project_id=project_id, if_exists="append"
  )
