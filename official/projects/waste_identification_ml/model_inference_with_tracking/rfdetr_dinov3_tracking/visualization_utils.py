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

"""Rendering, plotting, video writing, and summary utilities."""

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import math
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image
import tqdm

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import config_loader

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


@dataclasses.dataclass(frozen=True)
class TrackSummary:
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

  def __getitem__(self, item: str) -> Any:
    return getattr(self, item)


class PipelineVisualizer:
  """Manages file I/O, frame annotation, grid generation, and summary printing."""

  def __init__(
      self,
      config: config_loader.VisualizationConfig,
      collapsed_categories: config_loader.CollapsedCategoriesConfig,
      out_video_path: str,
      summary_logger: logging.Logger | None = None,
  ):
    """Initializes the visualizer.

    Args:
        config: Visualization-specific configuration.
        collapsed_categories: Optional grouping of fine-grained classes into
          broader categories. When disabled, per-category reporting and folder
          nesting are skipped.
        out_video_path: Filesystem path where the annotated MP4 will be written
          when `save_video` is true.
        summary_logger: Optional dedicated logger whose handler writes the
          per-subfolder summary blocks to a single summary file. When provided,
          every summary line is emitted to both the main logger
          (console/per-subfolder log) and this logger. When None, summaries are
          logged only to the main logger.
    """
    self._config = config
    self._collapsed_categories = collapsed_categories
    self._box_annotator = supervision.BoxAnnotator()
    self._label_annotator = supervision.LabelAnnotator()
    self._video_path = out_video_path
    self._video_writer = None
    self._summary_logger = summary_logger

  def annotate_and_write_frame(
      self, image: Image.Image, detections: Any, frame_path: str
  ) -> None:
    """Draws bounding boxes onto the frame and saves based on config toggles."""
    if not self._config.save_frames and not self._config.save_video:
      return

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated = self._box_annotator.annotate(
        scene=frame_bgr.copy(), detections=detections
    )

    labels = self._build_labels(detections)
    annotated = self._label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    if self._config.save_frames:
      cv2.imwrite(frame_path, annotated)

    if self._config.save_video:
      if self._video_writer is None:
        h, w = annotated.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            self._video_path, fourcc, self._config.output_video_fps, (w, h)
        )
      self._video_writer.write(annotated)

  def close_video(self) -> None:
    """Releases the video writer buffer."""
    if self._video_writer:
      self._video_writer.release()

  def save_track_grids(
      self,
      track_predictions: Mapping[int, Sequence[Mapping[str, Any]]],
      resolve_fn: Callable[[Sequence[Mapping[str, Any]]], tuple[str, int]],
      out_dir: str,
  ) -> dict[int, TrackSummary]:
    """Resolves labels and renders high-speed NumPy/OpenCV prediction grids.

    Output folder structure depends on whether collapsed categories are
    enabled:
        - Enabled:
        out_dir/<category>/<final_class>/track_NNNN_<final_class>.png
        - Disabled: out_dir/<final_class>/track_NNNN_<final_class>.png

    Args:
        track_predictions: Mapping from tracker_id to its per-crop prediction
          list.
        resolve_fn: Callable returning (final_class, vote_count) for a list of
          per-crop predictions.
        out_dir: Root directory under which track-grid PNGs are saved.

    Returns:
        Mapping from tracker_id to a TrackSummary dataclass containing
        `final_class`, `category`, `vote_count`, and `output_path`.
        `category` is None when the feature is disabled.
    """
    summary: dict[int, TrackSummary] = {}
    sorted_ids = sorted(track_predictions.keys())

    desc = (
        "Saving track grids"
        if self._config.save_track_grids
        else "Resolving track labels"
    )
    progress_bar = tqdm.tqdm(sorted_ids, desc=desc, unit="track")

    # Pre-calculate grid dimensions based on config (convert inches to pixels)
    tile_size = int(
        self._config.track_grid_thumbnail_size_inches
        * self._config.track_grid_dpi
    )
    img_size = int(tile_size * 0.75)  # 75% of tile for image, 25% for text
    header_height = 80
    cols = self._config.track_grid_columns_per_row

    for tracker_id in progress_bar:
      per_crop_preds = track_predictions[tracker_id]
      if not per_crop_preds:
        continue

      final_class, votes = resolve_fn(per_crop_preds)
      progress_bar.set_postfix_str(f"track {tracker_id:04d} -> {final_class}")

      category = self._collapsed_categories.get_category_for_class(final_class)
      path = "N/A (grids disabled)"

      if self._config.save_track_grids:
        count = len(per_crop_preds)
        num_cols = min(count, cols)
        num_rows = math.ceil(count / num_cols)

        # Create a blank white canvas for the entire grid
        grid_w = num_cols * tile_size
        grid_h = num_rows * tile_size + header_height
        grid_canvas = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

        # Draw the title header
        title = f"Track {tracker_id} | final: {final_class} ({count} crops)"
        cv2.putText(
            grid_canvas,
            title,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Stamp each crop onto the canvas
        for idx, pred in enumerate(per_crop_preds):
          r_idx = idx // num_cols
          c_idx = idx % num_cols

          x_offset = c_idx * tile_size
          y_offset = header_height + (r_idx * tile_size)

          # Create individual white tile
          tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

          # Convert PIL RGB crop to OpenCV BGR and resize it
          crop_bgr = cv2.cvtColor(np.array(pred["crop"]), cv2.COLOR_RGB2BGR)
          crop_resized = cv2.resize(
              crop_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR
          )

          # Center the image horizontally in the tile
          img_x_offset = (tile_size - img_size) // 2
          tile[10 : img_size + 10, img_x_offset : img_x_offset + img_size] = (
              crop_resized
          )

          # Draw text annotations below the image
          text_y = img_size + 40
          cv2.putText(
              tile,
              pred["frame_name"],
              (15, text_y),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (0, 0, 0),
              1,
              cv2.LINE_AA,
          )
          cv2.putText(
              tile,
              pred["predicted_class"],
              (15, text_y + 25),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.45,
              (0, 0, 0),
              1,
              cv2.LINE_AA,
          )
          cv2.putText(
              tile,
              f"{pred['predicted_probability_percent']:.2f}%",
              (15, text_y + 50),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (0, 0, 0),
              1,
              cv2.LINE_AA,
          )

          # Stamp the tile into the main grid canvas
          grid_canvas[
              y_offset : y_offset + tile_size, x_offset : x_offset + tile_size
          ] = tile

        class_dir = self._build_class_output_directory(
            out_dir, category, final_class
        )
        os.makedirs(class_dir, exist_ok=True)
        path = os.path.join(
            class_dir, f"track_{tracker_id:04d}_{final_class}.png"
        )

        # Save the final numpy array directly to disk
        cv2.imwrite(path, grid_canvas)

      summary[tracker_id] = TrackSummary(
          final_class=final_class,
          category=category,
          vote_count=votes,
          output_path=path,
      )

    progress_bar.close()
    return summary

  def _log_summary_line(self, message: str, *args) -> None:
    """Logs one summary line to both the main logger and the summary file.

    The message is always emitted through the main logger (console and the
    per-subfolder log file). When a summary logger was supplied, the same
    fully-formatted line is also written to the dedicated summary file.

    Args:
        message: A printf-style logging message.
        *args: Arguments interpolated into `message`.
    """
    _LOGGER.info(message, *args)
    if self._summary_logger is not None:
      self._summary_logger.info(message % args if args else message)

  def print_summary(
      self,
      track_summary: Mapping[int, TrackSummary | Mapping[str, Any]],
      input_directory: str,
      class_names: list[str],
  ) -> None:
    """Logs per-class object counts and ground-truth class accuracy.

    Every line is written both to the main logger (console and the
    per-subfolder log) and, when a summary logger was supplied at
    construction time, to the dedicated summary file. A blank separator
    line is written to the summary file before each block so stacked
    per-subfolder blocks stay readable.

    When collapsed categories are enabled, also logs per-category counts
    and ground-truth category accuracy. The ground-truth category is
    derived from the ground-truth class via the collapsed-category
    mapping, not matched from the subfolder name.

    Args:
        track_summary: Mapping from tracker_id to its resolved summary dict or
          TrackSummary.
        input_directory: Path to the subfolder that produced this summary.
        class_names: Full list of class labels from the config.
    """
    if self._summary_logger is not None:
      self._summary_logger.info("")

    class_counts: collections.Counter[str] = collections.Counter(
        s.final_class if isinstance(s, TrackSummary) else s["final_class"]
        for s in track_summary.values()
    )
    total_objects = sum(class_counts.values())

    ground_truth_class = self._infer_ground_truth_class(
        input_directory, class_names
    )

    self._log_summary_line("Folder name: %s", os.path.basename(input_directory))
    self._log_summary_line("Total tracked objects: %d", total_objects)

    self._log_summary_line("By class:")
    for class_name, count in class_counts.most_common():
      self._log_summary_line("  %s: %d", class_name, count)

    self._log_class_accuracy(
        class_counts=class_counts,
        total_objects=total_objects,
        ground_truth_class=ground_truth_class,
    )

    if self._collapsed_categories.enable:
      self._log_collapsed_category_section(
          track_summary=track_summary,
          total_objects=total_objects,
          ground_truth_class=ground_truth_class,
      )

  def _log_class_accuracy(
      self,
      class_counts: collections.Counter[str],
      total_objects: int,
      ground_truth_class: str | None,
  ) -> None:
    """Logs the ground-truth class accuracy line.

    Args:
      class_counts: Counts of tracked objects per final class.
      total_objects: Total number of tracked objects.
      ground_truth_class: The class inferred from the subfolder name by exact
        match, or None when it could not be inferred.
    """
    if ground_truth_class is None:
      self._log_summary_line(
          "  Class accuracy: N/A (could not infer ground-truth class from"
          " subfolder name)"
      )
    elif total_objects == 0:
      self._log_summary_line("  Class accuracy: N/A (no tracked objects)")
    else:
      class_accuracy = class_counts.get(ground_truth_class, 0) / total_objects
      self._log_summary_line(
          "  Class accuracy (vs %s): %.2f%%",
          ground_truth_class,
          class_accuracy * 100,
      )

  def _log_collapsed_category_section(
      self,
      track_summary: Mapping[int, TrackSummary | Mapping[str, Any]],
      total_objects: int,
      ground_truth_class: str | None,
  ) -> None:
    """Logs the 'By collapsed categories' counts and accuracy line.

    The ground-truth category is derived from the ground-truth class via
    the collapsed-category mapping (rather than matched from the subfolder
    name), so class and category ground truth always agree.

    Args:
        track_summary: Mapping from tracker_id to its resolved summary dict or
          TrackSummary.
        total_objects: Total number of tracked objects.
        ground_truth_class: The class inferred from the subfolder name, or None
          when it could not be inferred.
    """
    category_counts: collections.Counter[str] = collections.Counter(
        s.category if isinstance(s, TrackSummary) else s["category"]
        for s in track_summary.values()
    )

    self._log_summary_line("By collapsed categories:")
    for category_name, count in category_counts.most_common():
      self._log_summary_line("  %s: %d", category_name, count)

    ground_truth_category = self._derive_ground_truth_category(
        ground_truth_class
    )
    if ground_truth_category is None:
      self._log_summary_line(
          "  Category accuracy: N/A (could not infer ground-truth category "
          "from subfolder name)"
      )
    elif total_objects == 0:
      self._log_summary_line("  Category accuracy: N/A (no tracked objects)")
    else:
      category_accuracy = (
          category_counts.get(ground_truth_category, 0) / total_objects
      )
      self._log_summary_line(
          "  Category accuracy (vs %s): %.2f%%",
          ground_truth_category,
          category_accuracy * 100,
      )

  def _derive_ground_truth_category(
      self, ground_truth_class: str | None
  ) -> str | None:
    """Derives the ground-truth category from the ground-truth class.

    The category is looked up through the collapsed-category mapping
    rather than matched from the subfolder name, so a subfolder named
    exactly after a class (e.g. 'brown_bottles_grade3') yields both its
    class ground truth and, via the mapping, its category ground truth
    (e.g. 'grade3'). Returns None when the class ground truth could not
    be inferred.

    Args:
        ground_truth_class: The class inferred from the subfolder name, or None.

    Returns:
        The category that contains the ground-truth class, or None.
    """
    if ground_truth_class is None:
      return None
    return self._collapsed_categories.get_category_for_class(ground_truth_class)

  def _infer_ground_truth_class(
      self, input_directory: str, class_names: list[str]
  ) -> str | None:
    """Returns the configured class whose name exactly equals the subfolder name.

    Matching is a case-insensitive exact comparison against the basename
    of the input directory. If zero or multiple classes match, the result
    is ambiguous and None is returned.

    Args:
        input_directory: Path to the per-subfolder input directory.
        class_names: Full list of class labels from the config.

    Returns:
        The matched class name, or None if zero or multiple classes match.
    """
    return self._infer_ground_truth_from_names(input_directory, class_names)

  def _infer_ground_truth_from_names(
      self, input_directory: str, candidate_names: list[str]
  ) -> str | None:
    """Returns the candidate whose name exactly equals the subfolder basename.

    Matching is a case-insensitive exact comparison against the basename
    of the input directory: the subfolder must be named exactly after one
    of the candidate names. A subfolder whose name merely contains a
    candidate as a substring (e.g. 'brown_bottles_grade3_batch1') is not
    treated as a match. If zero or multiple candidates match, returns None
    to signal an ambiguous result.

    Args:
        input_directory: Path to the per-subfolder input directory.
        candidate_names: List of names to match against.

    Returns:
        The single exactly-matched name, or None if zero or multiple
        names match.
    """
    if not candidate_names:
      return None
    subfolder_name = os.path.basename(input_directory).lower()
    matched_names = [
        name for name in candidate_names if name.lower() == subfolder_name
    ]
    if len(matched_names) == 1:
      return matched_names[0]
    return None

  def _build_class_output_directory(
      self, out_dir: str, category: str | None, final_class: str
  ) -> str:
    """Returns the directory in which a track-grid PNG should be saved.

    Args:
        out_dir: Root output directory for track grids.
        category: The collapsed category for the final class, or None if the
          feature is disabled.
        final_class: The resolved fine-grained class for the track.

    Returns:
        Filesystem path to the directory where the grid should be saved.
    """
    if category is None:
      return os.path.join(out_dir, final_class)
    return os.path.join(out_dir, category, final_class)

  def _build_labels(self, detections: Any) -> list[str]:
    """Formats the display string over bounded objects."""
    if not detections:
      return []

    ids = (
        list(detections.tracker_id)
        if detections.tracker_id is not None
        else [None] * len(detections)
    )
    confs = (
        list(detections.confidence)
        if detections.confidence is not None
        else [None] * len(detections)
    )

    labels = []
    for tid, conf in zip(ids, confs):
      id_str = "?" if tid is None or int(tid) == -1 else f"ID {int(tid)}"
      if self._config.show_confidence_in_labels and conf is not None:
        labels.append(f"{id_str} {float(conf):.2f}")
      else:
        labels.append(id_str)
    return labels
