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

"""Unit tests for the PipelineVisualizer public API."""

import logging
import pathlib
from unittest import mock

from absl.testing import absltest
import numpy
from PIL import Image

from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader
from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import visualization_utils


def _make_visualization_config(
    save_frames: bool = True,
    save_video: bool = True,
    save_track_grids: bool = True,
) -> config_loader.VisualizationConfig:
  """Returns a VisualizationConfig with values suitable for tests."""
  return config_loader.VisualizationConfig(
      save_frames=save_frames,
      save_video=save_video,
      save_track_grids=save_track_grids,
      output_video_fps=30,
      show_confidence_in_labels=True,
      background_blend_color_rgb=(0, 0, 0),
      track_grid_columns_per_row=2,
      track_grid_thumbnail_size_inches=2,
      track_grid_dpi=100,
  )


def _make_collapsed_categories_disabled() -> (
    config_loader.CollapsedCategoriesConfig
):
  """Returns a disabled CollapsedCategoriesConfig."""
  return config_loader.CollapsedCategoriesConfig(enable=False, mapping={})


def _make_collapsed_categories_enabled() -> (
    config_loader.CollapsedCategoriesConfig
):
  """Returns an enabled CollapsedCategoriesConfig with two categories."""
  return config_loader.CollapsedCategoriesConfig(
      enable=True,
      mapping={
          "recyclable": ["bottle", "can"],
          "trash": ["wrapper"],
      },
  )


def _make_per_crop_prediction(
    predicted_class: str = "bottle",
    predicted_probability_percent: float = 85.5,
    frame_name: str = "frame_0001.png",
) -> visualization_utils.PerCropPrediction:
  """Returns a PerCropPrediction backed by a small solid-color PIL image."""
  return visualization_utils.PerCropPrediction(
      crop=Image.new("RGB", (16, 16), color="red"),
      frame_name=frame_name,
      predicted_class=predicted_class,
      predicted_probability_percent=predicted_probability_percent,
  )


class AnnotateAndWriteFrameTest(absltest.TestCase):
  """Tests for PipelineVisualizer.annotate_and_write_frame."""

  def setUp(self):
    """Patches supervision annotators and cv2 calls used during annotation."""
    super().setUp()
    self.enter_context(
        mock.patch.object(
            visualization_utils.supervision,
            "BoxAnnotator",
            autospec=True,
            create=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            visualization_utils.supervision,
            "LabelAnnotator",
            autospec=True,
            create=True,
        )
    )
    self.mock_cvt_color = self.enter_context(
        mock.patch.object(
            visualization_utils.cv2, "cvtColor", autospec=False, create=True
        )
    )
    self.mock_imwrite = self.enter_context(
        mock.patch.object(
            visualization_utils.cv2, "imwrite", autospec=False, create=True
        )
    )
    self.mock_video_writer_class = self.enter_context(
        mock.patch.object(
            visualization_utils.cv2,
            "VideoWriter",
            autospec=False,
            create=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            visualization_utils.cv2,
            "VideoWriter_fourcc",
            autospec=False,
            create=True,
            return_value=0,
        )
    )

    # cvtColor returns a synthetic BGR frame with known shape.

    self.mock_cvt_color.return_value = numpy.zeros(
        (10, 20, 3), dtype=numpy.uint8
    )
    # Both annotators return their scene unchanged.
    box_annotator_instance = (
        visualization_utils.supervision.BoxAnnotator.return_value
    )
    box_annotator_instance.annotate.side_effect = lambda scene, detections: (
        scene
    )
    label_annotator_instance = (
        visualization_utils.supervision.LabelAnnotator.return_value
    )
    label_annotator_instance.annotate.side_effect = (
        lambda scene, detections, labels: scene
    )

  def _make_visualizer(
      self, save_frames: bool = True, save_video: bool = True
  ) -> visualization_utils.PipelineVisualizer:
    """Returns a PipelineVisualizer with categories disabled."""
    return visualization_utils.PipelineVisualizer(
        config=_make_visualization_config(
            save_frames=save_frames, save_video=save_video
        ),
        collapsed_categories=_make_collapsed_categories_disabled(),
        output_video_path=pathlib.Path("/tmp/video.mp4"),
    )

  def _make_detections(self) -> mock.MagicMock:
    """Returns a fake detections object satisfying visualizer expectations."""
    detections = mock.MagicMock()
    detections.__len__.return_value = 1
    detections.tracker_id = numpy.array([1])
    detections.confidence = numpy.array([0.9])
    return detections

  def test_skips_all_work_when_both_toggles_off(self):
    """Verifies no annotation or I/O occurs when both save toggles are false."""
    visualizer = self._make_visualizer(save_frames=False, save_video=False)
    visualizer.annotate_and_write_frame(
        image=Image.new("RGB", (20, 10), color="red"),
        detections=self._make_detections(),
        frame_path=pathlib.Path("/tmp/frame.png"),
    )
    self.mock_cvt_color.assert_not_called()
    self.mock_imwrite.assert_not_called()
    self.mock_video_writer_class.assert_not_called()

  def test_writes_frame_when_save_frames_enabled(self):
    """Verifies imwrite is called with the string form of the frame path."""
    visualizer = self._make_visualizer(save_frames=True, save_video=False)
    visualizer.annotate_and_write_frame(
        image=Image.new("RGB", (20, 10), color="red"),
        detections=self._make_detections(),
        frame_path=pathlib.Path("/tmp/frame.png"),
    )
    self.mock_imwrite.assert_called_once()
    called_path = self.mock_imwrite.call_args.args[0]
    self.assertEqual(called_path, "/tmp/frame.png")
    self.mock_video_writer_class.assert_not_called()

  def test_opens_video_writer_lazily_on_first_frame(self):
    """Verifies the video writer is constructed lazily on the first frame."""
    visualizer = self._make_visualizer(save_frames=False, save_video=True)
    video_writer_instance = self.mock_video_writer_class.return_value

    visualizer.annotate_and_write_frame(
        image=Image.new("RGB", (20, 10), color="red"),
        detections=self._make_detections(),
        frame_path=pathlib.Path("/tmp/frame.png"),
    )
    self.mock_video_writer_class.assert_called_once()
    # Width and height come from the annotated BGR frame shape (10, 20, 3).
    _, _, _, size = self.mock_video_writer_class.call_args.args
    self.assertEqual(size, (20, 10))
    video_writer_instance.write.assert_called_once()

    # A second frame should reuse the same writer instance.
    visualizer.annotate_and_write_frame(
        image=Image.new("RGB", (20, 10), color="red"),
        detections=self._make_detections(),
        frame_path=pathlib.Path("/tmp/frame_2.png"),
    )
    self.mock_video_writer_class.assert_called_once()
    self.assertEqual(video_writer_instance.write.call_count, 2)


class CloseVideoTest(absltest.TestCase):
  """Tests for PipelineVisualizer.close_video."""

  def test_release_is_a_no_op_when_no_video_was_written(self):
    """Verifies close_video does not fail when no writer was opened."""
    visualizer = visualization_utils.PipelineVisualizer(
        config=_make_visualization_config(),
        collapsed_categories=_make_collapsed_categories_disabled(),
        output_video_path=pathlib.Path("/tmp/video.mp4"),
    )
    # Should not raise.
    visualizer.close_video()

  def test_release_is_called_when_writer_exists(self):
    """Verifies close_video releases the underlying writer."""
    visualizer = visualization_utils.PipelineVisualizer(
        config=_make_visualization_config(),
        collapsed_categories=_make_collapsed_categories_disabled(),
        output_video_path=pathlib.Path("/tmp/video.mp4"),
    )
    mock_writer = mock.MagicMock()
    visualizer._video_writer = mock_writer
    visualizer.close_video()
    mock_writer.release.assert_called_once()
    self.assertIsNone(visualizer._video_writer)


class SaveTrackGridsTest(absltest.TestCase):
  """Tests for PipelineVisualizer.save_track_grids."""

  def setUp(self):
    """Patches cv2 write operations and tqdm progress-bar iteration."""
    super().setUp()
    self.mock_imwrite = self.enter_context(
        mock.patch.object(visualization_utils.cv2, "imwrite", autospec=True)
    )
    # tqdm wraps the iterable; we make it behave like a plain iterator with
    # no-op set_postfix_str / close methods.
    self.enter_context(
        mock.patch.object(
            visualization_utils.tqdm,
            "tqdm",
            autospec=True,
            side_effect=lambda iterable, **kwargs: _FakeProgressBar(iterable),
        )
    )

  def _make_visualizer(

      self,
      save_track_grids: bool,
      collapsed_categories: config_loader.CollapsedCategoriesConfig,
  ) -> visualization_utils.PipelineVisualizer:
    """Returns a PipelineVisualizer with the given grid-saving toggle."""
    return visualization_utils.PipelineVisualizer(
        config=_make_visualization_config(save_track_grids=save_track_grids),
        collapsed_categories=collapsed_categories,
        output_video_path=pathlib.Path("/tmp/video.mp4"),
    )

  def test_returns_summary_and_skips_disk_when_grids_disabled(self):
    """Verifies labels are resolved but no PNGs are written when disabled."""
    visualizer = self._make_visualizer(
        save_track_grids=False,
        collapsed_categories=_make_collapsed_categories_disabled(),
    )
    track_predictions = {
        1: [_make_per_crop_prediction(predicted_class="bottle")],
    }
    resolve_labels = mock.MagicMock(return_value=("bottle", 1))

    summary = visualizer.save_track_grids(
        track_predictions=track_predictions,
        resolve_labels=resolve_labels,
        output_directory=pathlib.Path("/tmp/grids"),
    )

    self.assertEqual(summary[1]["final_class"], "bottle")
    self.assertIsNone(summary[1]["category"])
    self.assertEqual(summary[1]["vote_count"], 1)
    self.assertEqual(summary[1]["output_path"], "N/A (grids disabled)")
    self.mock_imwrite.assert_not_called()

  def test_skips_tracks_with_no_predictions(self):
    """Verifies tracker_ids with empty prediction lists are not included."""
    visualizer = self._make_visualizer(
        save_track_grids=False,
        collapsed_categories=_make_collapsed_categories_disabled(),
    )
    track_predictions = {
        1: [],
        2: [_make_per_crop_prediction()],
    }
    resolve_labels = mock.MagicMock(return_value=("bottle", 1))

    summary = visualizer.save_track_grids(
        track_predictions=track_predictions,
        resolve_labels=resolve_labels,
        output_directory=pathlib.Path("/tmp/grids"),
    )
    self.assertNotIn(1, summary)
    self.assertIn(2, summary)

  def test_writes_png_under_flat_directory_when_categories_disabled(self):
    """Verifies output path is <root>/<final_class>/track_NNNN_<class>.png."""
    visualizer = self._make_visualizer(
        save_track_grids=True,
        collapsed_categories=_make_collapsed_categories_disabled(),
    )
    track_predictions = {
        7: [_make_per_crop_prediction(predicted_class="bottle")],
    }
    resolve_labels = mock.MagicMock(return_value=("bottle", 1))

    with mock.patch.object(
        visualization_utils.pathlib.Path, "mkdir", autospec=True
    ):
      summary = visualizer.save_track_grids(
          track_predictions=track_predictions,
          resolve_labels=resolve_labels,
          output_directory=pathlib.Path("/tmp/grids"),
      )

    self.assertEqual(
        summary[7]["output_path"], "/tmp/grids/bottle/track_0007_bottle.png"
    )
    self.mock_imwrite.assert_called_once()

  def test_writes_png_under_category_directory_when_categories_enabled(self):
    """Verifies output path is <root>/<category>/<final_class>/track_...png."""
    visualizer = self._make_visualizer(
        save_track_grids=True,
        collapsed_categories=_make_collapsed_categories_enabled(),
    )
    track_predictions = {
        3: [_make_per_crop_prediction(predicted_class="bottle")],
    }
    resolve_labels = mock.MagicMock(return_value=("bottle", 1))

    with mock.patch.object(
        visualization_utils.pathlib.Path, "mkdir", autospec=True
    ):
      summary = visualizer.save_track_grids(
          track_predictions=track_predictions,
          resolve_labels=resolve_labels,
          output_directory=pathlib.Path("/tmp/grids"),
      )

    self.assertEqual(summary[3]["category"], "recyclable")
    self.assertEqual(
        summary[3]["output_path"],
        "/tmp/grids/recyclable/bottle/track_0003_bottle.png",
    )


class PrintSummaryTest(absltest.TestCase):
  """Tests for PipelineVisualizer.print_summary."""

  def _make_visualizer(
      self,
      collapsed_categories: config_loader.CollapsedCategoriesConfig,
  ) -> visualization_utils.PipelineVisualizer:
    """Returns a PipelineVisualizer with categories as specified."""
    return visualization_utils.PipelineVisualizer(
        config=_make_visualization_config(),
        collapsed_categories=collapsed_categories,
        output_video_path=pathlib.Path("/tmp/video.mp4"),
    )

  def _make_summary(
      self,
  ) -> dict[int, visualization_utils.TrackSummary]:
    """Returns a small summary with two bottle tracks and one wrapper track."""
    return {
        1: visualization_utils.TrackSummary(
            final_class="bottle",
            category="recyclable",
            vote_count=5,
            output_path="/tmp/a.png",
        ),
        2: visualization_utils.TrackSummary(
            final_class="bottle",
            category="recyclable",
            vote_count=4,
            output_path="/tmp/b.png",
        ),
        3: visualization_utils.TrackSummary(
            final_class="wrapper",
            category="trash",
            vote_count=3,
            output_path="/tmp/c.png",
        ),
    }

  def test_logs_total_and_per_class_counts(self):
    """Verifies total and per-class lines are logged."""
    visualizer = self._make_visualizer(
        collapsed_categories=_make_collapsed_categories_disabled()
    )
    with self.assertLogs(
        visualization_utils._LOGGER.name, level=logging.INFO
    ) as logs:
      visualizer.print_summary(
          track_summary=self._make_summary(),
          input_directory=pathlib.Path("/data/bottle_session"),
          class_names=["bottle", "wrapper", "can"],
      )
    joined = "\n".join(logs.output)
    self.assertIn("Total tracked objects: 3", joined)
    self.assertIn("bottle: 2", joined)
    self.assertIn("wrapper: 1", joined)

  def test_logs_class_accuracy_when_ground_truth_inferable(self):
    """Verifies class accuracy is computed when the subfolder name matches."""
    visualizer = self._make_visualizer(
        collapsed_categories=_make_collapsed_categories_disabled()
    )
    with self.assertLogs(
        visualization_utils._LOGGER.name, level=logging.INFO
    ) as logs:
      visualizer.print_summary(
          track_summary=self._make_summary(),
          input_directory=pathlib.Path("/data/bottle_session"),
          class_names=["bottle", "wrapper", "can"],
      )
    joined = "\n".join(logs.output)
    # 2 of 3 tracks are 'bottle' -> 66.67%.
    self.assertIn("Class accuracy (vs bottle): 66.67%", joined)

  def test_reports_class_accuracy_na_when_subfolder_unmatched(self):
    """Verifies a clear message when no class name matches the subfolder."""
    visualizer = self._make_visualizer(
        collapsed_categories=_make_collapsed_categories_disabled()
    )
    with self.assertLogs(
        visualization_utils._LOGGER.name, level=logging.INFO
    ) as logs:
      visualizer.print_summary(
          track_summary=self._make_summary(),
          input_directory=pathlib.Path("/data/unrelated_folder"),
          class_names=["bottle", "wrapper", "can"],
      )
    self.assertIn("Class accuracy: N/A", "\n".join(logs.output))

  def test_logs_category_section_when_categories_enabled(self):
    """Verifies the category section is present when collapsed cats are on."""
    visualizer = self._make_visualizer(
        collapsed_categories=_make_collapsed_categories_enabled()
    )
    with self.assertLogs(
        visualization_utils._LOGGER.name, level=logging.INFO
    ) as logs:
      visualizer.print_summary(
          track_summary=self._make_summary(),
          input_directory=pathlib.Path("/data/recyclable_session"),
          class_names=["bottle", "wrapper", "can"],
      )
    joined = "\n".join(logs.output)
    self.assertIn("By collapsed categories:", joined)
    self.assertIn("recyclable: 2", joined)
    self.assertIn("trash: 1", joined)
    # 2 of 3 tracks are 'recyclable' -> 66.67%.
    self.assertIn("Category accuracy (vs recyclable): 66.67%", joined)

  def test_omits_category_section_when_categories_disabled(self):
    """Verifies the category section is absent when collapsed cats are off."""
    visualizer = self._make_visualizer(
        collapsed_categories=_make_collapsed_categories_disabled()
    )
    with self.assertLogs(
        visualization_utils._LOGGER.name, level=logging.INFO
    ) as logs:
      visualizer.print_summary(
          track_summary=self._make_summary(),
          input_directory=pathlib.Path("/data/bottle_session"),
          class_names=["bottle", "wrapper", "can"],
      )
    self.assertNotIn("By collapsed categories:", "\n".join(logs.output))

  def test_infer_ground_truth_from_names_returns_none_when_candidates_empty(
      self,
  ):
    """Verifies _infer_ground_truth_from_names returns None if candidates empty."""
    self.assertIsNone(
        visualization_utils._infer_ground_truth_from_names(
            input_directory=pathlib.Path("/data/bottle_session"),
            candidate_names=[],
        )
    )


class SupervisionFallbackTest(absltest.TestCase):
  """Tests for _SupervisionFallback dummy classes."""

  def test_fallback_annotators_return_scene_unchanged(self):
    """Verifies fallback BoxAnnotator and LabelAnnotator return scene as-is."""
    if not hasattr(visualization_utils, "_SupervisionFallback"):
      self.skipTest("supervision package is installed; fallback not active.")
    fallback = visualization_utils._SupervisionFallback()
    box_annotator = fallback.BoxAnnotator()
    label_annotator = fallback.LabelAnnotator()
    dummy_scene = numpy.zeros((10, 10, 3), dtype=numpy.uint8)

    self.assertIs(
        box_annotator.annotate(dummy_scene, detections=None),
        dummy_scene,
    )
    self.assertIs(
        label_annotator.annotate(dummy_scene, detections=None, labels=[]),
        dummy_scene,
    )


class _FakeProgressBar:
  """Minimal stand-in for tqdm.tqdm used in save_track_grids tests."""

  def __init__(self, iterable):
    self._iterable = list(iterable)

  def __iter__(self):
    return iter(self._iterable)

  def set_postfix_str(self, text: str) -> None:
    del text
    return None

  def close(self) -> None:
    return None


if __name__ == "__main__":
  absltest.main()
