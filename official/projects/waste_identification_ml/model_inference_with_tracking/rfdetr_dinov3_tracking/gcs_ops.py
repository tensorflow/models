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

"""Google Cloud Platform operations for the pipeline.

Provides Cloud Storage transfers and BigQuery result ingestion.
"""

from collections.abc import Mapping
import logging
import os
import shutil
from typing import Any

from google.cloud import bigquery
from google.cloud import exceptions
from google.cloud import storage

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import config_loader

_LOGGER = logging.getLogger(__name__)

_GCS_URI_PREFIX = "gs://"

# Column order for the per-track table. This is also the column order for the
# BigQuery table, so it is kept in one place.
TRACK_TABLE_COLUMNS = (
    "session_name",
    "tracker_id",
    "final_class",
    "vote_count",
    "collapsed_class",
    "track_grid_filename",
)

_TRACK_GRID_NOT_AVAILABLE = "N/A"

# BigQuery table schema as (column_name, field_type) pairs. The column names
# and their order match TRACK_TABLE_COLUMNS. All fields are NULLABLE (the
# default mode), so empty values such as a missing collapsed class ingest as
# NULL. Kept as plain tuples so importing this module does not require the
# BigQuery client; it is converted to SchemaField objects at ingestion time.
_BIGQUERY_SCHEMA = (
    ("session_name", "STRING"),
    ("tracker_id", "INTEGER"),
    ("final_class", "STRING"),
    ("vote_count", "INTEGER"),
    ("collapsed_class", "STRING"),
    ("track_grid_filename", "STRING"),
)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
  """Splits a gs:// URI into its bucket name and object prefix.

  Args:
      gcs_uri: A URI of the form 'gs://bucket-name/optional/prefix/'.

  Returns:
      A tuple of (bucket_name, object_prefix). The object prefix is an
      empty string when the URI points at the bucket root.

  Raises:
      ValueError: If the URI does not start with 'gs://' or has no bucket
          name.
  """
  if not gcs_uri.startswith(_GCS_URI_PREFIX):
    raise ValueError(f"GCS URI must start with '{_GCS_URI_PREFIX}': {gcs_uri}")
  uri_without_scheme = gcs_uri[len(_GCS_URI_PREFIX) :]
  bucket_name, _, object_prefix = uri_without_scheme.partition("/")
  if not bucket_name:
    raise ValueError(f"GCS URI has no bucket name: {gcs_uri}")
  return bucket_name, object_prefix


def resolve_io_directories(
    paths: config_loader.PathsConfig,
) -> tuple[str, str]:
  """Resolves the working input/output directories for the active source.

  In local mode the configured local directories are returned unchanged.
  In GCS mode the input is downloaded from the configured input URI into
  the temp input directory, and the temp input/output directories are
  returned as the working directories. The local branch does not construct
  a Cloud Storage client, so no GCP credentials are needed when running
  fully local.

  Args:
      paths: The validated paths configuration. Exactly one of its local or GCS
        sources is enabled.

  Returns:
      A tuple of (input_root, output_root) that the pipeline reads from
      and writes to.
  """
  if paths.gcs.enable:
    _LOGGER.info("GCS mode: downloading input from %s ...", paths.gcs.input_uri)
    storage_manager = CloudStorageManager()
    storage_manager.download_directory(
        gcs_uri=paths.gcs.input_uri,
        local_directory=paths.gcs.temp_input_directory,
    )
    return (
        paths.gcs.temp_input_directory,
        paths.gcs.temp_output_directory,
    )
  return (
      paths.local.input_image_directory,
      paths.local.output_root_directory,
  )


def upload_output_directory(paths: config_loader.PathsConfig) -> None:
  """Uploads pipeline output to GCS when GCS mode is active.

  In local mode this is a no-op, so callers can invoke it unconditionally.
  In GCS mode the entire temp output directory is uploaded to the
  configured output URI, preserving folder structure.

  Args:
      paths: The validated paths configuration. Exactly one of its local or GCS
        sources is enabled.
  """
  if not paths.gcs.enable:
    return

  _LOGGER.info("GCS mode: uploading output to %s ...", paths.gcs.output_uri)
  storage_manager = CloudStorageManager()
  storage_manager.upload_directory(
      local_directory=paths.gcs.temp_output_directory,
      gcs_uri=paths.gcs.output_uri,
  )


def build_track_rows(
    track_summary: Mapping[Any, Any], session_name: str
) -> list[dict[str, Any]]:
  """Builds one table row per track from a subfolder's track summary.

  The row shape is the BigQuery payload. `final_class` is the DINOv3 grade
  resolved for the track, `collapsed_class` is its optional category (None
  when collapsed categories are disabled), and `track_grid_filename` is the
  saved grid PNG name ('N/A' when grids are disabled).

  Args:
      track_summary: Mapping from tracker_id to its resolved summary dict, as
        returned by PipelineVisualizer.save_track_grids. Each value contains
        'final_class', 'category', 'vote_count', and 'output_path'.
      session_name: Name of the capture session (the subfolder name).

  Returns:
      A list of row dicts keyed by TRACK_TABLE_COLUMNS, ordered by
      tracker_id.
  """
  rows = []
  for tracker_id in sorted(track_summary):
    summary = track_summary[tracker_id]
    rows.append({
        "session_name": session_name,
        "tracker_id": int(tracker_id),
        "final_class": summary["final_class"],
        "vote_count": int(summary["vote_count"]),
        "collapsed_class": summary["category"],
        "track_grid_filename": _track_grid_filename(summary["output_path"]),
    })
  return rows


def print_track_rows(rows: list[dict[str, Any]]) -> None:
  """Prints track rows as a simple table for reviewing the schema.

  Temporary scaffolding used to review the BigQuery schema.

  Args:
      rows: Row dicts as produced by build_track_rows.
  """
  print(f"Schema: {' | '.join(TRACK_TABLE_COLUMNS)}")
  for row in rows:
    values = [str(row[column]) for column in TRACK_TABLE_COLUMNS]
    print(" | ".join(values))


def _track_grid_filename(output_path: str) -> str:
  """Returns the track-grid PNG filename, or 'N/A' when grids are disabled.

  When track grids are disabled, PipelineVisualizer.save_track_grids stores
  a placeholder string rather than a real path. That case is reported as
  'N/A'; otherwise the basename of the saved PNG is returned.

  Args:
      output_path: The 'output_path' field from a track summary entry.

  Returns:
      The PNG filename (e.g. 'track_0011_dirt_jars_grade3.png') or 'N/A'.
  """
  if not output_path or output_path.startswith(_TRACK_GRID_NOT_AVAILABLE):
    return _TRACK_GRID_NOT_AVAILABLE
  return os.path.basename(output_path)


def ingest_track_rows(
    bigquery_config: config_loader.BigQueryConfig,
    rows: list[dict[str, Any]],
) -> None:
  """Writes track rows to BigQuery when BigQuery ingestion is enabled.

  This runs independently of the input/output source, so results can be
  written whether the pipeline ran in local or GCS mode. It is a no-op when
  BigQuery is disabled or when there are no rows, so callers can invoke it
  unconditionally.

  Args:
      bigquery_config: Validated BigQuery configuration.
      rows: Track rows as produced by build_track_rows, accumulated across all
        processed subfolders.
  """
  if not bigquery_config.enable:
    return
  if not rows:
    _LOGGER.info("BigQuery enabled but there are no rows to ingest.")
    return

  _LOGGER.info("Ingesting %d row(s) into BigQuery ...", len(rows))
  manager = BigQueryManager(
      project_id=bigquery_config.project_id,
      dataset_id=bigquery_config.dataset_id,
      table_id=bigquery_config.table_id,
  )
  manager.ingest_rows(rows, overwrite=bigquery_config.overwrite)


class CloudStorageManager:
  """Handles Cloud Storage transfers for pipeline input and output."""

  def __init__(self, project_id: str | None = None):
    """Initializes the Cloud Storage client.

    Args:
        project_id: Google Cloud project ID. When None, the client uses the
          project inferred from the environment credentials.
    """
    self._client = storage.Client(project=project_id)

  def download_directory(self, gcs_uri: str, local_directory: str) -> int:
    """Downloads all objects under a GCS prefix into a local directory.

    The local directory is cleared before downloading so that stale data
    from a previous run does not remain. The last segment of the prefix
    (the named folder) is preserved locally, so the folder itself is
    recreated rather than having its contents flattened.

    Args:
        gcs_uri: Source URI of the form 'gs://bucket/prefix/'.
        local_directory: Local destination directory. Created if missing,
          emptied if it already exists.

    Returns:
        The number of files downloaded.

    Raises:
        ValueError: If the GCS URI is malformed.
    """
    bucket_name, object_prefix = _parse_gcs_uri(gcs_uri)
    self._reset_local_directory(local_directory)

    bucket = self._client.bucket(bucket_name)
    blobs = list(self._client.list_blobs(bucket, prefix=object_prefix))

    downloaded_count = 0
    for blob in blobs:
      # Skip directory placeholder objects whose name ends with a slash.
      if blob.name.endswith("/"):
        continue
      relative_path = self._relative_object_path(
          blob_name=blob.name, object_prefix=object_prefix
      )
      destination_path = os.path.join(local_directory, relative_path)
      os.makedirs(os.path.dirname(destination_path), exist_ok=True)
      blob.download_to_filename(destination_path)
      downloaded_count += 1

    _LOGGER.info(
        "Downloaded %d file(s) from %s to %s",
        downloaded_count,
        gcs_uri,
        local_directory,
    )
    return downloaded_count

  def upload_directory(self, local_directory: str, gcs_uri: str) -> int:
    """Uploads every file under a local directory to a GCS prefix.

    The local directory tree is walked recursively and each file is
    uploaded to the destination prefix, preserving the relative folder
    structure. Existing objects at the destination are overwritten.

    Args:
        local_directory: Local source directory to upload.
        gcs_uri: Destination URI of the form 'gs://bucket/prefix/'.

    Returns:
        The number of files uploaded.

    Raises:
        ValueError: If the GCS URI is malformed.
    """
    bucket_name, object_prefix = _parse_gcs_uri(gcs_uri)
    bucket = self._client.bucket(bucket_name)

    uploaded_count = 0
    for current_directory, _, file_names in os.walk(local_directory):
      for file_name in file_names:
        local_path = os.path.join(current_directory, file_name)
        relative_path = os.path.relpath(local_path, local_directory)
        blob_name = self._join_gcs_path(object_prefix, relative_path)
        bucket.blob(blob_name).upload_from_filename(local_path)
        uploaded_count += 1

    _LOGGER.info(
        "Uploaded %d file(s) from %s to %s",
        uploaded_count,
        local_directory,
        gcs_uri,
    )
    return uploaded_count

  def _reset_local_directory(self, local_directory: str) -> None:
    """Removes and recreates a local directory so it starts empty.

    Args:
        local_directory: Directory to reset.
    """
    if os.path.isdir(local_directory):
      shutil.rmtree(local_directory)
    os.makedirs(local_directory, exist_ok=True)

  def _relative_object_path(self, blob_name: str, object_prefix: str) -> str:
    """Returns a blob's path relative to the parent of the download prefix.

    The last segment of the prefix (the named folder, e.g. 'images') is
    preserved so that the folder itself is recreated locally rather than
    having its contents flattened into the destination directory. For a
    prefix 'a/b/images/' and blob 'a/b/images/img1.png', this returns
    'images/img1.png'.

    Args:
        blob_name: Full object name within the bucket.
        object_prefix: The prefix the download was rooted at.

    Returns:
        The portion of the blob name below the prefix's parent, suitable
        for joining onto the local destination directory.
    """
    parent_prefix = self._parent_prefix(object_prefix)
    relative_path = blob_name[len(parent_prefix) :]
    return relative_path.lstrip("/")

  def _parent_prefix(self, object_prefix: str) -> str:
    """Returns the parent portion of a prefix, keeping its last folder.

    For 'a/b/images/' this returns 'a/b/'. For a single-segment prefix
    like 'images/' this returns '' so the folder is kept at the top of
    the destination. An empty prefix (bucket root) returns ''.

    Args:
        object_prefix: The prefix the download was rooted at.

    Returns:
        The prefix with its last folder segment removed.
    """
    trimmed_prefix = object_prefix.rstrip("/")
    if "/" not in trimmed_prefix:
      return ""
    parent, _, _ = trimmed_prefix.rpartition("/")
    return parent + "/"

  def _join_gcs_path(self, object_prefix: str, relative_path: str) -> str:
    """Joins a GCS prefix and a local relative path into a blob name.

    GCS object names always use forward slashes, so any OS-specific
    separators in the relative path are normalized.

    Args:
        object_prefix: Destination prefix (may be empty for bucket root).
        relative_path: File path relative to the upload source directory.

    Returns:
        The full blob name under which the file should be stored.
    """
    normalized_relative = relative_path.replace(os.sep, "/")
    clean_prefix = object_prefix.strip("/")
    if not clean_prefix:
      return normalized_relative
    return clean_prefix + "/" + normalized_relative


class BigQueryManager:
  """Handles BigQuery dataset/table setup and result ingestion.

  The BigQuery client is imported lazily so that importing this module does
  not require google-cloud-bigquery to be installed when BigQuery is not
  used.
  """

  def __init__(self, project_id: str, dataset_id: str, table_id: str):
    """Initializes the BigQuery client and resolves the table reference.

    Args:
        project_id: Google Cloud project ID that owns the dataset.
        dataset_id: BigQuery dataset ID that holds the table.
        table_id: BigQuery table ID that receives the rows.
    """
    self._bigquery = bigquery
    self._client = bigquery.Client(project=project_id)
    self._project_id = project_id
    self._dataset_id = dataset_id
    self._table_id = table_id
    self._table_reference = f"{project_id}.{dataset_id}.{table_id}"

  def ingest_rows(self, rows: list[dict[str, Any]], overwrite: bool) -> None:
    """Loads rows into the table, creating the dataset/table if needed.

    The table is created from the module schema when it does not already
    exist. When overwrite is True the existing rows are replaced; when
    False the new rows are appended.

    Args:
        rows: Row dicts keyed by the schema column names.
        overwrite: Whether to replace existing rows instead of appending.
    """
    self._ensure_dataset()

    schema = [
        self._bigquery.SchemaField(column_name, field_type)
        for column_name, field_type in _BIGQUERY_SCHEMA
    ]
    if overwrite:
      write_disposition = self._bigquery.WriteDisposition.WRITE_TRUNCATE
    else:
      write_disposition = self._bigquery.WriteDisposition.WRITE_APPEND

    job_config = self._bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=write_disposition,
    )
    load_job = self._client.load_table_from_json(
        rows, self._table_reference, job_config=job_config
    )
    load_job.result()
    _LOGGER.info("Ingested %d row(s) into %s", len(rows), self._table_reference)

  def _ensure_dataset(self) -> None:
    """Creates the dataset if it does not already exist."""
    dataset_reference = self._bigquery.DatasetReference(
        self._project_id, self._dataset_id
    )
    try:
      self._client.get_dataset(dataset_reference)
    except exceptions.NotFound:
      _LOGGER.info("Dataset %s not found. Creating ...", self._dataset_id)
      self._client.create_dataset(self._bigquery.Dataset(dataset_reference))
