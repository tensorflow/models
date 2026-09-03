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

"""Unit tests for gcs_ops.py.

The Cloud Storage and BigQuery clients are patched so no network calls or
credentials are needed. The pure URI/path/row helpers are exercised directly.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import gcs_ops


class ParseGcsUriTest(parameterized.TestCase):
  """Tests for _parse_gcs_uri."""

  @parameterized.named_parameters(
      ("bucket_and_prefix", "gs://bucket/a/b/", ("bucket", "a/b/")),
      ("bucket_root", "gs://bucket/", ("bucket", "")),
      ("bucket_no_slash", "gs://bucket", ("bucket", "")),
      ("nested_object", "gs://bkt/x/y/z.png", ("bkt", "x/y/z.png")),
  )
  def test_splits_bucket_and_prefix(self, uri, expected):
    """Verifies the bucket and object prefix are split correctly."""
    self.assertEqual(gcs_ops._parse_gcs_uri(uri), expected)

  def test_raises_without_scheme(self):
    """Verifies a URI missing the gs:// scheme raises ValueError."""
    with self.assertRaises(ValueError):
      gcs_ops._parse_gcs_uri("bucket/path/")

  def test_raises_without_bucket(self):
    """Verifies a URI with no bucket name raises ValueError."""
    with self.assertRaises(ValueError):
      gcs_ops._parse_gcs_uri("gs:///path/")


class TrackGridFilenameTest(parameterized.TestCase):
  """Tests for _track_grid_filename."""

  def test_returns_basename_for_real_path(self):
    """Verifies a real path is reduced to its basename."""
    self.assertEqual(
        gcs_ops._track_grid_filename("/out/grade3/track_0011_dirt.png"),
        "track_0011_dirt.png",
    )

  @parameterized.named_parameters(
      ("empty", ""),
      ("na_marker", "N/A (grids disabled)"),
  )
  def test_returns_na_for_placeholder(self, output_path):
    """Verifies an empty or placeholder path reports 'N/A'."""
    self.assertEqual(gcs_ops._track_grid_filename(output_path), "N/A")


class BuildTrackRowsTest(absltest.TestCase):
  """Tests for build_track_rows."""

  def test_builds_one_row_per_track_sorted_by_id(self):
    """Verifies rows are produced per track, ordered by tracker_id."""
    track_summary = {
        2: {
            "final_class": "dirt_jars_grade3",
            "category": "grade3",
            "vote_count": 4,
            "output_path": "/o/grade3/track_0002_dirt_jars_grade3.png",
        },
        1: {
            "final_class": "clean_jars_grade1",
            "category": "grade1",
            "vote_count": 7,
            "output_path": "/o/grade1/track_0001_clean_jars_grade1.png",
        },
    }
    rows = gcs_ops.build_track_rows(track_summary, session_name="session_a")
    self.assertEqual([row["tracker_id"] for row in rows], [1, 2])
    self.assertEqual(rows[0]["session_name"], "session_a")
    self.assertEqual(rows[0]["final_class"], "clean_jars_grade1")
    self.assertEqual(rows[0]["collapsed_class"], "grade1")
    self.assertEqual(
        rows[0]["track_grid_filename"], "track_0001_clean_jars_grade1.png"
    )

  def test_row_keys_match_table_columns(self):
    """Verifies every row carries exactly the schema columns."""
    track_summary = {
        0: {
            "final_class": "c",
            "category": None,
            "vote_count": 1,
            "output_path": "N/A (grids disabled)",
        }
    }
    rows = gcs_ops.build_track_rows(track_summary, session_name="s")
    self.assertCountEqual(rows[0].keys(), gcs_ops.TRACK_TABLE_COLUMNS)

  def test_disabled_grid_reports_na_filename(self):
    """Verifies a placeholder output path maps to an 'N/A' filename."""
    track_summary = {
        0: {
            "final_class": "c",
            "category": None,
            "vote_count": 1,
            "output_path": "N/A (grids disabled)",
        }
    }
    rows = gcs_ops.build_track_rows(track_summary, session_name="s")
    self.assertEqual(rows[0]["track_grid_filename"], "N/A")
    self.assertIsNone(rows[0]["collapsed_class"])


class CloudStorageManagerPathHelpersTest(parameterized.TestCase):
  """Tests for the pure path helpers on CloudStorageManager.

  The storage client is patched so constructing the manager does not touch
  GCP; only the string-manipulation helpers are exercised.
  """

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(gcs_ops.storage, "Client", autospec=True)
    )
    self.manager = gcs_ops.CloudStorageManager()

  @parameterized.named_parameters(
      ("nested_prefix", "a/b/images/", "a/b/"),
      ("single_segment", "images/", ""),
      ("empty_prefix", "", ""),
  )
  def test_parent_prefix(self, object_prefix, expected):
    """Verifies the parent prefix keeps the prefix's final folder."""
    self.assertEqual(self.manager._parent_prefix(object_prefix), expected)

  def test_relative_object_path_preserves_last_folder(self):
    """Verifies a blob path keeps the named download folder locally."""
    result = self.manager._relative_object_path(
        blob_name="a/b/images/img1.png", object_prefix="a/b/images/"
    )
    self.assertEqual(result, "images/img1.png")

  @parameterized.named_parameters(
      ("with_prefix", "out/run", "a/b.png", "out/run/a/b.png"),
      ("empty_prefix", "", "a/b.png", "a/b.png"),
      ("prefix_trailing_slash", "out/", "b.png", "out/b.png"),
  )
  def test_join_gcs_path(self, object_prefix, relative_path, expected):
    """Verifies prefix and relative path join into a forward-slash blob name."""
    self.assertEqual(
        self.manager._join_gcs_path(object_prefix, relative_path), expected
    )


class ResolveIoDirectoriesTest(absltest.TestCase):
  """Tests for resolve_io_directories (local branch needs no GCP client)."""

  def _paths(self, local_enable: bool, gcs_enable: bool):
    """Builds a minimal PathsConfig-like stub with local and gcs sub-objects."""
    paths = mock.Mock()
    paths.local = mock.Mock(
        enable=local_enable,
        input_image_directory="/data/in",
        output_root_directory="/data/out",
    )
    paths.gcs = mock.Mock(
        enable=gcs_enable,
        input_uri="gs://b/in/",
        output_uri="gs://b/out/",
        temp_input_directory="/tmp/in",
        temp_output_directory="/tmp/out",
    )
    return paths

  def test_local_mode_returns_configured_dirs(self):
    """Verifies local mode returns the configured local directories."""
    paths = self._paths(local_enable=True, gcs_enable=False)
    result = gcs_ops.resolve_io_directories(paths)
    self.assertEqual(result, ("/data/in", "/data/out"))

  def test_gcs_mode_downloads_and_returns_temp_dirs(self):
    """Verifies GCS mode downloads input and returns the temp directories."""
    paths = self._paths(local_enable=False, gcs_enable=True)
    with mock.patch.object(
        gcs_ops, "CloudStorageManager", autospec=True
    ) as mock_manager_class:
      result = gcs_ops.resolve_io_directories(paths)
      mock_manager_class.return_value.download_directory.assert_called_once()
    self.assertEqual(result, ("/tmp/in", "/tmp/out"))


class UploadOutputDirectoryTest(absltest.TestCase):
  """Tests for upload_output_directory."""

  def test_local_mode_is_noop(self):
    """Verifies local mode does not construct a storage manager."""
    paths = mock.Mock()
    paths.gcs = mock.Mock(enable=False)
    with mock.patch.object(
        gcs_ops, "CloudStorageManager", autospec=True
    ) as mock_manager_class:
      gcs_ops.upload_output_directory(paths)
      mock_manager_class.assert_not_called()

  def test_gcs_mode_uploads(self):
    """Verifies GCS mode uploads the temp output directory."""
    paths = mock.Mock()
    paths.gcs = mock.Mock(
        enable=True,
        output_uri="gs://b/out/",
        temp_output_directory="/tmp/out",
    )
    with mock.patch.object(
        gcs_ops, "CloudStorageManager", autospec=True
    ) as mock_manager_class:
      gcs_ops.upload_output_directory(paths)
      mock_manager_class.return_value.upload_directory.assert_called_once()


class IngestTrackRowsTest(absltest.TestCase):
  """Tests for ingest_track_rows."""

  def test_disabled_is_noop(self):
    """Verifies a disabled BigQuery config skips ingestion."""
    config = mock.Mock(enable=False)
    with mock.patch.object(
        gcs_ops, "BigQueryManager", autospec=True
    ) as mock_manager_class:
      gcs_ops.ingest_track_rows(config, rows=[{"a": 1}])
      mock_manager_class.assert_not_called()

  def test_enabled_with_no_rows_is_noop(self):
    """Verifies an empty row list skips ingestion even when enabled."""
    config = mock.Mock(
        enable=True, project_id="p", dataset_id="d", table_id="t"
    )
    with mock.patch.object(
        gcs_ops, "BigQueryManager", autospec=True
    ) as mock_manager_class:
      gcs_ops.ingest_track_rows(config, rows=[])
      mock_manager_class.assert_not_called()

  def test_enabled_with_rows_ingests(self):
    """Verifies enabled ingestion constructs the manager and writes rows."""
    config = mock.Mock(
        enable=True,
        project_id="p",
        dataset_id="d",
        table_id="t",
        overwrite=True,
    )
    rows = [{"session_name": "s"}]
    with mock.patch.object(
        gcs_ops, "BigQueryManager", autospec=True
    ) as mock_manager_class:
      gcs_ops.ingest_track_rows(config, rows=rows)
      mock_manager_class.assert_called_once_with(
          project_id="p", dataset_id="d", table_id="t"
      )
      mock_manager_class.return_value.ingest_rows.assert_called_once_with(
          rows, overwrite=True
      )


if __name__ == "__main__":
  absltest.main()
