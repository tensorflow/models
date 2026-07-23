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

"""Rendering, plotting, video writing, and summary utilities."""

import collections
from collections.abc import Callable, Mapping, Sequence
import logging
import math
import pathlib
from typing import TypedDict

import cv2
import numpy
from PIL import Image
import tqdm

from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import config_loader

try:
  import supervision  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:

  class _SupervisionFallback:
    """Fallback classes when supervision is not installed."""

    class BoxAnnotator:

      def annotate(self, scene, detections):
        del detections
        return scene

    class LabelAnnotator:

      def annotate(self, scene, detections, labels):
        del detections, labels
        return scene

    class Detections:

      def __len__(self) -> int:
        return 0

  supervision = _SupervisionFallback()


_LOGGER = logging.getLogger(__name__)
# Rendering constants for track-grid PNG output.
# Image occupies this fraction of a tile; the remainder is used for text.
_TILE_IMAGE_RATIO = 0.75
_GRID_HEADER_HEIGHT_PIXELS = 80
_IMAGE_TOP_PADDING_PIXELS = 10
_TILE_TEXT_TOP_OFFSET_PIXELS = 40
_TILE_TEXT_LINE_SPACING_PIXELS = 25
_TILE_TEXT_LARGE_LINE_SPACING_PIXELS = 50
_TILE_TEXT_LEFT_PADDING_PIXELS = 15
_HEADER_TEXT_LEFT_PADDING_PIXELS = 20
_HEADER_TEXT_BASELINE_PIXELS = 50
_HEADER_FONT_SCALE = 1.0
_HEADER_FONT_THICKNESS = 2
_TILE_FONT_SCALE_STANDARD = 0.5
_TILE_FONT_SCALE_SMALL = 0.45
_TILE_FONT_THICKNESS = 1
_TEXT_COLOR_BGR = (0, 0, 0)
_CANVAS_COLOR_BGR = (255, 255, 255)
_VIDEO_FOURCC = "mp4v"


class PerCropPrediction(TypedDict):
  """Per-crop prediction record consumed by the track-grid renderer.

  Attributes:
    crop: The PIL RGB crop that was classified.
    frame_name: Name of the frame the crop came from.
    predicted_class: Top-1 predicted class name.
    predicted_probability_percent: Top-1 probability as a percentage in [0.0,
      100.0].
  """

  crop: Image.Image
  frame_name: str
  predicted_class: str
  predicted_probability_percent: float


class TrackSummary(TypedDict):
  """Resolved summary of a single tracker's predictions.

  Attributes:
    final_class: The class chosen after aggregating per-crop predictions.
    category: The collapsed category for `final_class`, or None when the
      collapsed-category feature is disabled.
    vote_count: Number of per-crop predictions that voted for `final_class`.
    output_path: Filesystem path where the track-grid PNG was saved, or a
      placeholder string when grid saving is disabled.
  """

  final_class: str
  category: str | None
  vote_count: int
  output_path: str


# Signature of the callable that resolves per-crop predictions into a single
# (final_class, vote_count) result for a tracker.
LabelResolver = Callable[[Sequence[PerCropPrediction]], tuple[str, int]]


class PipelineVisualizer:
  """Manages frame annotation, grid generation, video writing, and summaries."""

  def __init__(
      self,
      config: config_loader.VisualizationConfig,
      collapsed_categories: config_loader.CollapsedCategoriesConfig,
      output_video_path: pathlib.Path,
  ):
    """Initializes the visualizer.

    Args:
      config: Visualization-specific configuration.
      collapsed_categories: Optional grouping of fine-grained classes into
        broader categories. When disabled, per-category reporting and folder
        nesting are skipped.
      output_video_path: Filesystem path where the annotated MP4 will be written
        when `save_video` is true.
    """
    self._config = config
    self._collapsed_categories = collapsed_categories
    self._box_annotator = supervision.BoxAnnotator()
    self._label_annotator = supervision.LabelAnnotator()
    self._output_video_path = output_video_path
    self._video_writer: cv2.VideoWriter | None = None

  def annotate_and_write_frame(
      self,
      image: Image.Image,
      detections: supervision.Detections,
      frame_path: pathlib.Path,
  ) -> None:
    """Annotates a frame and writes it to disk and/or video per the config.

    Args:
      image: Source frame as a PIL RGB image.
      detections: Detections to overlay on the frame.
      frame_path: Filesystem path where the annotated frame PNG will be saved
        when `save_frames` is true.
    """
    if not self._config.save_frames and not self._config.save_video:
      return

    annotated_frame = self._annotate_frame(image=image, detections=detections)

    if self._config.save_frames:
      cv2.imwrite(str(frame_path), annotated_frame)

    if self._config.save_video:
      self._write_video_frame(annotated_frame)

  def close_video(self) -> None:
    """Releases the video writer buffer if one was opened."""
    if self._video_writer is not None:
      self._video_writer.release()
      self._video_writer = None

  def save_track_grids(
      self,
      track_predictions: Mapping[int, Sequence[PerCropPrediction]],
      resolve_labels: LabelResolver,
      output_directory: pathlib.Path,
  ) -> dict[int, TrackSummary]:
    """Resolves labels and renders per-track prediction grids.

    Output folder structure depends on whether collapsed categories are
    enabled:
      - Enabled:  <output_directory>/<category>/<final_class>/
          track_NNNN_<final_class>.png
      - Disabled: <output_directory>/<final_class>/
          track_NNNN_<final_class>.png

    Args:
      track_predictions: Mapping from tracker_id to its per-crop prediction
        list.
      resolve_labels: Callable returning (final_class, vote_count) for a list of
        per-crop predictions.
      output_directory: Root directory under which track-grid PNGs are saved.

    Returns:
      Mapping from tracker_id to a TrackSummary. The `category` field is
      None when the collapsed-category feature is disabled.
    """
    summary: dict[int, TrackSummary] = {}
    sorted_tracker_ids = sorted(track_predictions.keys())

    progress_description = (
        "Saving track grids"
        if self._config.save_track_grids
        else "Resolving track labels"
    )
    progress_bar = tqdm.tqdm(
        sorted_tracker_ids, desc=progress_description, unit="track"
    )

    for tracker_id in progress_bar:
      per_crop_predictions = track_predictions[tracker_id]
      if not per_crop_predictions:
        continue

      final_class, vote_count = resolve_labels(per_crop_predictions)
      progress_bar.set_postfix_str(f"track {tracker_id:04d} -> {final_class}")

      category = self._collapsed_categories.get_category_for_class(final_class)
      output_path_string = "N/A (grids disabled)"

      if self._config.save_track_grids:
        output_path = self._render_and_save_track_grid(
            tracker_id=tracker_id,
            final_class=final_class,
            category=category,
            per_crop_predictions=per_crop_predictions,
            output_directory=output_directory,
        )
        output_path_string = str(output_path)

      summary[tracker_id] = TrackSummary(
          final_class=final_class,
          category=category,
          vote_count=vote_count,
          output_path=output_path_string,
      )

    progress_bar.close()
    return summary

  def print_summary(
      self,
      track_summary: Mapping[int, TrackSummary],
      input_directory: pathlib.Path,
      class_names: Sequence[str],
  ) -> None:
    """Logs per-class object counts and, if enabled, per-category counts.

    Also logs class accuracy and, when collapsed categories are enabled,
    category accuracy against the ground truth inferred from the input
    subfolder name.

    Args:
      track_summary: Mapping from tracker_id to its resolved summary.
      input_directory: Path to the subfolder that produced this summary.
      class_names: Full list of class labels from the config.
    """
    class_counts = collections.Counter(
        entry["final_class"] for entry in track_summary.values()
    )
    total_objects = sum(class_counts.values())

    _LOGGER.info("Folder name: %s", input_directory.name)
    _LOGGER.info("Total tracked objects: %d", total_objects)

    _LOGGER.info("By class:")
    for class_name, count in class_counts.most_common():
      _LOGGER.info("  %s: %d", class_name, count)

    self._log_class_accuracy(
        class_counts=class_counts,
        total_objects=total_objects,
        input_directory=input_directory,
        class_names=class_names,
    )

    if self._collapsed_categories.enable:
      self._log_collapsed_category_section(
          track_summary=track_summary,
          total_objects=total_objects,
          input_directory=input_directory,
      )

  def _annotate_frame(
      self, image: Image.Image, detections: supervision.Detections
  ) -> numpy.ndarray:
    """Returns a BGR annotated frame with boxes and labels drawn on it."""
    frame_bgr = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    annotated = self._box_annotator.annotate(
        scene=frame_bgr.copy(), detections=detections
    )
    labels = self._build_labels(detections)
    return self._label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

  def _write_video_frame(self, annotated_frame: numpy.ndarray) -> None:
    """Writes a BGR frame to the video, opening the writer if not yet open."""
    if self._video_writer is None:
      height, width = annotated_frame.shape[:2]
      fourcc = cv2.VideoWriter_fourcc(*_VIDEO_FOURCC)
      self._video_writer = cv2.VideoWriter(
          str(self._output_video_path),
          fourcc,
          self._config.output_video_fps,
          (width, height),
      )
    self._video_writer.write(annotated_frame)

  def _render_and_save_track_grid(
      self,
      tracker_id: int,
      final_class: str,
      category: str | None,
      per_crop_predictions: Sequence[PerCropPrediction],
      output_directory: pathlib.Path,
  ) -> pathlib.Path:
    """Renders one track grid PNG to disk and returns the output path."""
    tile_size_pixels = int(
        self._config.track_grid_thumbnail_size_inches
        * self._config.track_grid_dpi
    )
    image_size_pixels = int(tile_size_pixels * _TILE_IMAGE_RATIO)

    grid_canvas = self._build_grid_canvas(
        tracker_id=tracker_id,
        final_class=final_class,
        per_crop_predictions=per_crop_predictions,
        tile_size_pixels=tile_size_pixels,
        image_size_pixels=image_size_pixels,
    )

    class_directory = _build_class_output_directory(
        output_directory=output_directory,
        category=category,
        final_class=final_class,
    )
    class_directory.mkdir(parents=True, exist_ok=True)
    output_path = class_directory / f"track_{tracker_id:04d}_{final_class}.png"
    cv2.imwrite(str(output_path), grid_canvas)
    return output_path

  def _build_grid_canvas(
      self,
      tracker_id: int,
      final_class: str,
      per_crop_predictions: Sequence[PerCropPrediction],
      tile_size_pixels: int,
      image_size_pixels: int,
  ) -> numpy.ndarray:
    """Builds the grid canvas with a header and per-crop tiles drawn on it."""
    crop_count = len(per_crop_predictions)
    columns = min(crop_count, self._config.track_grid_columns_per_row)
    rows = math.ceil(crop_count / columns)

    grid_width_pixels = columns * tile_size_pixels
    grid_height_pixels = rows * tile_size_pixels + _GRID_HEADER_HEIGHT_PIXELS
    grid_canvas = numpy.full(
        (grid_height_pixels, grid_width_pixels, 3),
        _CANVAS_COLOR_BGR[0],
        dtype=numpy.uint8,
    )

    header_text = (
        f"Track {tracker_id} | final: {final_class} ({crop_count} crops)"
    )
    cv2.putText(
        grid_canvas,
        header_text,
        (_HEADER_TEXT_LEFT_PADDING_PIXELS, _HEADER_TEXT_BASELINE_PIXELS),
        cv2.FONT_HERSHEY_SIMPLEX,
        _HEADER_FONT_SCALE,
        _TEXT_COLOR_BGR,
        _HEADER_FONT_THICKNESS,
        cv2.LINE_AA,
    )

    for tile_index, prediction in enumerate(per_crop_predictions):
      tile = self._build_tile(
          prediction=prediction,
          tile_size_pixels=tile_size_pixels,
          image_size_pixels=image_size_pixels,
      )
      row_index = tile_index // columns
      column_index = tile_index % columns
      x_offset = column_index * tile_size_pixels
      y_offset = _GRID_HEADER_HEIGHT_PIXELS + row_index * tile_size_pixels
      grid_canvas[
          y_offset : y_offset + tile_size_pixels,
          x_offset : x_offset + tile_size_pixels,
      ] = tile

    return grid_canvas

  def _build_tile(
      self,
      prediction: PerCropPrediction,
      tile_size_pixels: int,
      image_size_pixels: int,
  ) -> numpy.ndarray:
    """Builds a single tile containing a resized crop and text annotations."""
    tile = numpy.full(
        (tile_size_pixels, tile_size_pixels, 3),
        _CANVAS_COLOR_BGR[0],
        dtype=numpy.uint8,
    )

    crop_bgr = cv2.cvtColor(numpy.array(prediction["crop"]), cv2.COLOR_RGB2BGR)
    crop_resized = cv2.resize(
        crop_bgr,
        (image_size_pixels, image_size_pixels),
        interpolation=cv2.INTER_LINEAR,
    )

    image_x_offset = (tile_size_pixels - image_size_pixels) // 2
    image_bottom = _IMAGE_TOP_PADDING_PIXELS + image_size_pixels
    tile[
        _IMAGE_TOP_PADDING_PIXELS:image_bottom,
        image_x_offset : image_x_offset + image_size_pixels,
    ] = crop_resized

    text_baseline_y = image_size_pixels + _TILE_TEXT_TOP_OFFSET_PIXELS
    cv2.putText(
        tile,
        prediction["frame_name"],
        (_TILE_TEXT_LEFT_PADDING_PIXELS, text_baseline_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        _TILE_FONT_SCALE_STANDARD,
        _TEXT_COLOR_BGR,
        _TILE_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        prediction["predicted_class"],
        (
            _TILE_TEXT_LEFT_PADDING_PIXELS,
            text_baseline_y + _TILE_TEXT_LINE_SPACING_PIXELS,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        _TILE_FONT_SCALE_SMALL,
        _TEXT_COLOR_BGR,
        _TILE_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        f"{prediction['predicted_probability_percent']:.2f}%",
        (
            _TILE_TEXT_LEFT_PADDING_PIXELS,
            text_baseline_y + _TILE_TEXT_LARGE_LINE_SPACING_PIXELS,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        _TILE_FONT_SCALE_STANDARD,
        _TEXT_COLOR_BGR,
        _TILE_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return tile

  def _log_class_accuracy(
      self,
      class_counts: collections.Counter[str],
      total_objects: int,
      input_directory: pathlib.Path,
      class_names: Sequence[str],
  ) -> None:
    """Logs the ground-truth class accuracy line."""
    ground_truth_class = _infer_ground_truth_from_names(
        input_directory=input_directory, candidate_names=class_names
    )
    if ground_truth_class is None:
      _LOGGER.info(
          "  Class accuracy: N/A (could not infer ground-truth class from "
          "subfolder name)"
      )
      return
    if total_objects == 0:
      _LOGGER.info("  Class accuracy: N/A (no tracked objects)")
      return
    class_accuracy = class_counts.get(ground_truth_class, 0) / total_objects
    _LOGGER.info(
        "  Class accuracy (vs %s): %.2f%%",
        ground_truth_class,
        class_accuracy * 100,
    )

  def _log_collapsed_category_section(
      self,
      track_summary: Mapping[int, TrackSummary],
      total_objects: int,
      input_directory: pathlib.Path,
  ) -> None:
    """Logs the collapsed-category counts and accuracy line."""
    category_counts = collections.Counter(
        entry["category"] for entry in track_summary.values()
    )

    _LOGGER.info("By collapsed categories:")
    for category_name, count in category_counts.most_common():
      _LOGGER.info("  %s: %d", category_name, count)

    ground_truth_category = _infer_ground_truth_from_names(
        input_directory=input_directory,
        candidate_names=self._collapsed_categories.category_names,
    )
    if ground_truth_category is None:
      _LOGGER.info(
          "  Category accuracy: N/A (could not infer ground-truth category "
          "from subfolder name)"
      )
      return
    if total_objects == 0:
      _LOGGER.info("  Category accuracy: N/A (no tracked objects)")
      return
    category_accuracy = (
        category_counts.get(ground_truth_category, 0) / total_objects
    )
    _LOGGER.info(
        "  Category accuracy (vs %s): %.2f%%",
        ground_truth_category,
        category_accuracy * 100,
    )

  def _build_labels(self, detections: supervision.Detections) -> list[str]:
    """Formats the label strings shown over each detected object."""
    if not detections:
      return []

    if detections.tracker_id is not None:
      tracker_ids = list(detections.tracker_id)
    else:
      tracker_ids = [None] * len(detections)

    if detections.confidence is not None:
      confidences = list(detections.confidence)
    else:
      confidences = [None] * len(detections)

    labels: list[str] = []
    for tracker_id, confidence in zip(tracker_ids, confidences):
      if tracker_id is None or int(tracker_id) == -1:
        identifier_string = "?"
      else:
        identifier_string = f"ID {int(tracker_id)}"
      if self._config.show_confidence_in_labels and confidence is not None:
        labels.append(f"{identifier_string} {float(confidence):.2f}")
      else:
        labels.append(identifier_string)
    return labels


def _infer_ground_truth_from_names(
    input_directory: pathlib.Path, candidate_names: Sequence[str]
) -> str | None:
  """Returns the candidate whose name appears in the subfolder basename.

  Matching is a case-insensitive substring match against the basename of
  the input directory. If zero or multiple candidates match, returns None
  to signal an ambiguous result.

  Args:
    input_directory: Path to the per-subfolder input directory.
    candidate_names: Names to match against.

  Returns:
    The single matched name, or None if zero or multiple names match.
  """
  if not candidate_names:
    return None
  subfolder_name = input_directory.name.lower()
  matched_names = [
      name for name in candidate_names if name.lower() in subfolder_name
  ]
  if len(matched_names) == 1:
    return matched_names[0]
  return None


def _build_class_output_directory(
    output_directory: pathlib.Path,
    category: str | None,
    final_class: str,
) -> pathlib.Path:
  """Returns the directory in which a track-grid PNG should be saved."""
  if category is None:
    return output_directory / final_class
  return output_directory / category / final_class

