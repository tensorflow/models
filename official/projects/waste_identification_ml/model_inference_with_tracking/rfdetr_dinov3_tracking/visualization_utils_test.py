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

"""Unit tests for visualization_utils.py.

supervision's annotators are patched at construction so no real annotator is
built. Ground-truth inference, category derivation, label building, output
directory routing, and the grids-disabled resolve path are exercised
directly.
"""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import visualization_utils


class _FakeDetections:
  """Minimal stand-in for supervision.Detections used by _build_labels."""

  def __init__(self, tracker_id, confidence):
    self.tracker_id = tracker_id
    self.confidence = confidence

  def __len__(self) -> int:
    if self.tracker_id is not None:
      return len(self.tracker_id)
    if self.confidence is not None:
      return len(self.confidence)
    return 0


def _make_visualizer(
    show_confidence: bool = True,
    save_track_grids: bool = False,
    collapsed_categories=None,
) -> visualization_utils.PipelineVisualizer:
  """Builds a PipelineVisualizer with supervision annotators patched out."""
  config = mock.Mock(
      show_confidence_in_labels=show_confidence,
      save_frames=False,
      save_video=False,
      save_track_grids=save_track_grids,
      track_grid_thumbnail_size_inches=3,
      track_grid_dpi=150,
      track_grid_columns_per_row=5,
  )
  if collapsed_categories is None:
    collapsed_categories = mock.Mock(enable=False)
    collapsed_categories.get_category_for_class.return_value = None
  with mock.patch.object(
      visualization_utils.supervision, "BoxAnnotator", autospec=True
  ), mock.patch.object(
      visualization_utils.supervision, "LabelAnnotator", autospec=True
  ):
    return visualization_utils.PipelineVisualizer(
        config=config,
        collapsed_categories=collapsed_categories,
        out_video_path="/tmp/out.mp4",
        summary_logger=None,
    )


class InferGroundTruthFromNamesTest(parameterized.TestCase):
  """Tests for _infer_ground_truth_from_names."""

  def setUp(self):
    super().setUp()
    self.visualizer = _make_visualizer()

  def test_exact_case_insensitive_match(self):
    """Verifies an exact case-insensitive subfolder match is returned."""
    result = self.visualizer._infer_ground_truth_from_names(
        "/data/Brown_Bottles_Grade3", ["brown_bottles_grade3", "clean_grade1"]
    )
    self.assertEqual(result, "brown_bottles_grade3")

  def test_substring_is_not_a_match(self):
    """Verifies a name that merely contains a class is not matched."""
    result = self.visualizer._infer_ground_truth_from_names(
        "/data/brown_bottles_grade3_batch1", ["brown_bottles_grade3"]
    )
    self.assertIsNone(result)

  def test_no_candidates_returns_none(self):
    """Verifies an empty candidate list returns None."""
    result = self.visualizer._infer_ground_truth_from_names(
        "/data/anything", []
    )
    self.assertIsNone(result)

  def test_no_match_returns_none(self):
    """Verifies a subfolder matching no class returns None."""
    result = self.visualizer._infer_ground_truth_from_names(
        "/data/unknown_folder", ["brown_bottles_grade3"]
    )
    self.assertIsNone(result)


class DeriveGroundTruthCategoryTest(absltest.TestCase):
  """Tests for _derive_ground_truth_category."""

  def test_none_class_yields_none_category(self):
    """Verifies a None ground-truth class yields a None category."""
    visualizer = _make_visualizer()
    self.assertIsNone(visualizer._derive_ground_truth_category(None))

  def test_derives_category_via_mapping(self):
    """Verifies the category is looked up from the class via the mapping."""
    collapsed = mock.Mock(enable=True)
    collapsed.get_category_for_class.return_value = "grade3"
    visualizer = _make_visualizer(collapsed_categories=collapsed)
    result = visualizer._derive_ground_truth_category("brown_bottles_grade3")
    self.assertEqual(result, "grade3")
    collapsed.get_category_for_class.assert_called_once_with(
        "brown_bottles_grade3"
    )


class BuildClassOutputDirectoryTest(absltest.TestCase):
  """Tests for _build_class_output_directory."""

  def test_flat_layout_when_category_none(self):
    """Verifies output nests only by class when categories are disabled."""
    visualizer = _make_visualizer()
    result = visualizer._build_class_output_directory(
        "/out", category=None, final_class="clean_grade1"
    )
    self.assertEqual(result, "/out/clean_grade1")

  def test_nested_layout_when_category_present(self):
    """Verifies output nests by category then class when enabled."""
    visualizer = _make_visualizer()
    result = visualizer._build_class_output_directory(
        "/out", category="grade1", final_class="clean_grade1"
    )
    self.assertEqual(result, "/out/grade1/clean_grade1")


class BuildLabelsTest(absltest.TestCase):
  """Tests for _build_labels."""

  def test_empty_detections_yield_empty_labels(self):
    """Verifies no detections produce no labels."""
    visualizer = _make_visualizer()
    detections = _FakeDetections(tracker_id=[], confidence=[])
    self.assertEqual(visualizer._build_labels(detections), [])

  def test_includes_confidence_when_configured(self):
    """Verifies labels show the id and confidence when enabled."""
    visualizer = _make_visualizer(show_confidence=True)
    detections = _FakeDetections(tracker_id=[3], confidence=[0.9])
    labels = visualizer._build_labels(detections)
    self.assertEqual(labels, ["ID 3 0.90"])

  def test_omits_confidence_when_disabled(self):
    """Verifies labels show only the id when confidence is disabled."""
    visualizer = _make_visualizer(show_confidence=False)
    detections = _FakeDetections(tracker_id=[3], confidence=[0.9])
    labels = visualizer._build_labels(detections)
    self.assertEqual(labels, ["ID 3"])

  def test_unassigned_id_renders_question_mark(self):
    """Verifies a -1 tracker id renders as '?'."""
    visualizer = _make_visualizer(show_confidence=False)
    detections = _FakeDetections(tracker_id=[-1], confidence=[0.5])
    labels = visualizer._build_labels(detections)
    self.assertEqual(labels, ["?"])

  def test_none_tracker_ids_render_question_marks(self):
    """Verifies absent tracker ids render as '?' for every detection."""
    visualizer = _make_visualizer(show_confidence=False)
    detections = _FakeDetections(tracker_id=None, confidence=[0.5, 0.6])
    labels = visualizer._build_labels(detections)
    self.assertEqual(labels, ["?", "?"])


class SaveTrackGridsResolveOnlyTest(absltest.TestCase):
  """Tests for save_track_grids when grid rendering is disabled.

  With save_track_grids=False the method resolves labels and builds the
  summary without any OpenCV rendering, so it is exercised end to end here.
  """

  def test_builds_summary_without_rendering(self):
    """Verifies the summary carries resolved labels and an 'N/A' path."""
    collapsed = mock.Mock(enable=True)
    collapsed.get_category_for_class.return_value = "grade3"
    visualizer = _make_visualizer(
        save_track_grids=False, collapsed_categories=collapsed
    )

    track_predictions = {
        1: [{
            "predicted_class": "dirt_jars_grade3",
            "predicted_probability_percent": 90.0,
        }],
        2: [],  # Empty prediction list is skipped.
    }
    resolve_fn = mock.Mock(return_value=("dirt_jars_grade3", 5))

    summary = visualizer.save_track_grids(
        track_predictions, resolve_fn, out_dir="/out"
    )
    self.assertIn(1, summary)
    self.assertNotIn(2, summary)  # empty list produced no entry
    self.assertIsInstance(summary[1], visualization_utils.TrackSummary)
    self.assertEqual(summary[1].final_class, "dirt_jars_grade3")
    self.assertEqual(summary[1].category, "grade3")
    self.assertEqual(summary[1].vote_count, 5)
    self.assertEqual(summary[1].output_path, "N/A (grids disabled)")
    # Also verify mapping subscription for backward compatibility.
    self.assertEqual(summary[1]["final_class"], "dirt_jars_grade3")
    self.assertEqual(summary[1]["category"], "grade3")
    self.assertEqual(summary[1]["vote_count"], 5)
    self.assertEqual(summary[1]["output_path"], "N/A (grids disabled)")


class PrintSummaryTest(absltest.TestCase):
  """Tests for PipelineVisualizer.print_summary."""

  def test_print_summary_with_collapsed_categories(self):
    """Verifies print_summary logs without errors when categories are enabled."""
    collapsed = mock.Mock(enable=True)
    collapsed.get_category_for_class.return_value = "grade3"
    visualizer = _make_visualizer(collapsed_categories=collapsed)
    track_summary = {
        1: visualization_utils.TrackSummary(
            final_class="dirt_jars_grade3",
            category="grade3",
            vote_count=5,
            output_path="N/A",
        )
    }
    visualizer.print_summary(
        track_summary,
        input_directory="/data/dirt_jars_grade3",
        class_names=["dirt_jars_grade3", "clean_grade1"],
    )

  def test_print_summary_categories_disabled(self):
    """Verifies print_summary logs without errors when categories are disabled."""
    visualizer = _make_visualizer()
    track_summary = {
        1: visualization_utils.TrackSummary(
            final_class="dirt_jars_grade3",
            category=None,
            vote_count=5,
            output_path="N/A",
        )
    }
    visualizer.print_summary(
        track_summary,
        input_directory="/data/dirt_jars_grade3",
        class_names=["dirt_jars_grade3"],
    )


if __name__ == "__main__":
  absltest.main()
