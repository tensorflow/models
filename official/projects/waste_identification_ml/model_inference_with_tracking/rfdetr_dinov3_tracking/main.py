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

"""PET Bottle Grade Detection Pipeline Orchestrator."""

from collections.abc import Sequence
import gc
import glob
import logging
import os
import pathlib
import sys
from typing import Any

import cv2
from PIL import Image
import torch
import tqdm

from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import config_loader
from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import dinov3_classifier

try:
  from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import gcs_ops  # pylint: disable=g-import-not-at-top
except ImportError:
  gcs_ops = None

try:
  from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import rfdetr_detector  # pylint: disable=g-import-not-at-top
except ImportError:
  rfdetr_detector = None

try:
  from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import track_manager  # pylint: disable=g-import-not-at-top
except ImportError:
  track_manager = None

try:
  from official.projects.waste_identification_ml.model_inference_with_tracking.rfdetr_dinov3_tracking import visualization_utils  # pylint: disable=g-import-not-at-top
except ImportError:
  visualization_utils = None

_LOGGER_NAME = "waste_identification_pipeline"
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FILE_NAME = "pipeline.log"
_SUMMARY_FILE_NAME = "summary.txt"
_MEMORY_FLUSH_FRAME_INTERVAL = 10


def get_logger() -> logging.Logger:
  """Returns the shared, configured pipeline logger."""
  logger = logging.getLogger(_LOGGER_NAME)
  if logger.handlers:
    return logger
  logger.setLevel(logging.INFO)
  stream_handler = logging.StreamHandler(stream=sys.stdout)
  stream_handler.setFormatter(
      logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
  )
  logger.addHandler(stream_handler)
  logger.propagate = False
  return logger


def attach_file_handler(log_file_path: str) -> logging.FileHandler:
  """Attaches a file handler to the pipeline logger."""
  logger = get_logger()
  file_handler = logging.FileHandler(
      str(log_file_path), mode="w", encoding="utf-8"
  )
  file_handler.setLevel(logger.level)
  file_handler.setFormatter(
      logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
  )
  logger.addHandler(file_handler)
  return file_handler


def detach_file_handler(file_handler: logging.FileHandler) -> None:
  """Removes a file handler from the pipeline logger and closes it."""
  logger = get_logger()
  logger.removeHandler(file_handler)
  file_handler.close()


def create_summary_logger(summary_file_path: str) -> logging.Logger:
  """Creates a dedicated summary logger that writes to a file."""
  logger = logging.getLogger(f"summary_{summary_file_path}")
  logger.setLevel(logging.INFO)
  logger.propagate = False
  handler = logging.FileHandler(summary_file_path, mode="w", encoding="utf-8")
  handler.setFormatter(logging.Formatter("%(message)s"))
  logger.addHandler(handler)
  return logger


def close_summary_logger(logger: logging.Logger | None) -> None:
  """Closes all handlers on the summary logger."""
  if logger is None:
    return
  for handler in list(logger.handlers):
    logger.removeHandler(handler)
    handler.close()


def ensure_directories_exist(
    directories: Sequence[str | pathlib.Path],
) -> None:
  """Creates each directory in the given sequence if it does not already exist."""
  for directory in directories:
    if directory:
      os.makedirs(str(directory), exist_ok=True)


def load_and_resize_image(
    image_path: str | pathlib.Path, max_short_side: int
) -> Image.Image:
  """Loads an image via OpenCV and downscales it using Lanczos interpolation.

  Args:
    image_path: Path to the image file.
    max_short_side: Maximum size for the shorter dimension.

  Returns:
    The resized PIL Image.

  Raises:
    OSError: If OpenCV cannot read the image file.
  """
  bgr_image = cv2.imread(str(image_path))
  if bgr_image is None:
    raise OSError(
        f"OpenCV could not read the image at {image_path!s}. The file may be "
        "missing, corrupt, or in an unsupported format."
    )
  height, width = bgr_image.shape[:2]
  short_side = min(height, width)
  if short_side > max_short_side:
    scale = max_short_side / short_side
    new_w, new_h = int(width * scale), int(height * scale)
    bgr_image = cv2.resize(
        bgr_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
    )
  rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
  return Image.fromarray(rgb_image)


def flush_memory() -> None:
  """Runs garbage collection and empties CUDA cache if available."""
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


class PETBottlePipeline:
  """Orchestrates image processing, object tracking, and classification."""

  def __init__(self, config_path: str):
    """Initializes pipeline components based on the configuration file.

    Models are loaded once here and reused across all subfolders.

    Args:
      config_path: Path to the YAML configuration file.

    Raises:
      RuntimeError: If rfdetr_detector module is not available.
    """
    self._config = config_loader.PipelineConfig.from_yaml(config_path)
    self._summary_logger = None

    logger = get_logger()
    logger.info("Using device: %s", self._config.models.rfdetr.device)
    logger.info(
        "Loading RFDETR model from %s ...",
        self._config.models.rfdetr.checkpoint_path,
    )
    if rfdetr_detector is None:
      raise RuntimeError("rfdetr_detector module is not available.")
    self._detector = rfdetr_detector.RFDETRDetector(
        rfdetr_config=self._config.models.rfdetr,
        post_processing_config=self._config.post_processing,
    )
    logger.info("RFDETR model ready.")

    logger.info(
        "Loading DINOv3 classifier (%s) from %s ...",
        self._config.models.dinov3.model_name,
        self._config.models.dinov3.checkpoint_path,
    )
    self._classifier = dinov3_classifier.DINOv3Classifier.from_config(
        config=self._config.models.dinov3,
        class_names=self._config.classes,
        device=self._config.models.rfdetr.device,
    )
    logger.info("DINOv3 classifier ready.")

  def run(self) -> None:
    """Executes the pipeline over every immediate subfolder of the input directory.

    Raises:
      RuntimeError: If gcs_ops module is not available.
    """
    logger = get_logger()
    if gcs_ops is None:
      raise RuntimeError("gcs_ops module is not available.")
    input_root, output_root = gcs_ops.resolve_io_directories(self._config.paths)

    subfolders = self._discover_subfolders(input_root)
    if not subfolders:
      logger.warning(
          "No subfolders found in %s. Nothing to process.", input_root
      )
      return

    ensure_directories_exist([output_root])
    logger.info(
        "Found %d subfolder(s) to process under %s",
        len(subfolders),
        input_root,
    )

    summary_file_path = os.path.join(output_root, _SUMMARY_FILE_NAME)
    self._summary_logger = create_summary_logger(summary_file_path)
    try:
      if not self._config.tracking.enable:
        self._summary_logger.info(
            "Tracking disabled - no per-subfolder summaries were generated."
        )

      all_track_rows = []
      for subfolder_path in subfolders:
        subfolder_name = os.path.basename(subfolder_path)
        subfolder_output_dir = os.path.join(output_root, subfolder_name)
        try:
          subfolder_rows = self._process_subfolder(
              subfolder_path, subfolder_output_dir
          )
          all_track_rows.extend(subfolder_rows)
        except Exception:  # pylint: disable=broad-exception-caught
          logger.exception(
              "Subfolder '%s' failed. Skipping to next.",
              subfolder_name,
          )
          continue

      logger.info("Pipeline completed for all subfolders.")
      logger.info("Summary written to %s", summary_file_path)
      gcs_ops.upload_output_directory(self._config.paths)
      gcs_ops.ingest_track_rows(self._config.bigquery, all_track_rows)
    finally:
      close_summary_logger(self._summary_logger)
      self._summary_logger = None

  def _process_subfolder(
      self, input_subfolder: str, output_subfolder: str
  ) -> list[dict[str, Any]]:
    """Runs the per-frame loop and classification for a single subfolder.

    Tracks are independent per subfolder: a fresh TrackManager and
    PipelineVisualizer are instantiated for every call. A dedicated log
    file is attached for the duration of this subfolder's processing.

    Args:
        input_subfolder: Absolute path to the input subfolder containing image
          frames.
        output_subfolder: Absolute path to the output subfolder where results,
          logs, and intermediate artifacts will be written.

    Returns:
      The per-track rows for this subfolder, as produced by build_track_rows.
      Empty when the subfolder has no images.

    Raises:
      RuntimeError: If track_manager, visualization_utils, or gcs_ops is not
        available.
    """
    logger = get_logger()
    subfolder_name = os.path.basename(input_subfolder)

    frame_output_dir = os.path.join(
        output_subfolder, self._config.paths.output_frame_subfolder
    )
    grid_output_dir = os.path.join(
        output_subfolder, self._config.paths.track_grid_subfolder
    )
    video_output_path = os.path.join(
        output_subfolder, self._config.paths.output_video_filename
    )
    log_file_path = os.path.join(output_subfolder, _LOG_FILE_NAME)

    ensure_directories_exist([output_subfolder])
    directories_to_create = []
    if self._config.visualization.save_frames:
      directories_to_create.append(frame_output_dir)
    if self._config.visualization.save_track_grids:
      directories_to_create.append(grid_output_dir)
    ensure_directories_exist(directories_to_create)

    file_handler = attach_file_handler(log_file_path)
    try:
      logger.info("Processing subfolder: %s", subfolder_name)

      image_paths = self._collect_image_paths(input_subfolder)
      if not image_paths:
        logger.warning(
            "No images found in subfolder '%s'. Skipping.", subfolder_name
        )
        return []

      logger.info("Found %d image(s) in %s", len(image_paths), subfolder_name)

      if (
          track_manager is None
          or visualization_utils is None
          or gcs_ops is None
      ):
        raise RuntimeError(
            "track_manager, visualization_utils, and gcs_ops modules are"
            " required."
        )

      tracker = track_manager.TrackManager(
          tracking_config=self._config.tracking,
          cropping_config=self._config.cropping,
          vis_config=self._config.visualization,
      )
      tracker.reset()
      visualizer = visualization_utils.PipelineVisualizer(
          config=self._config.visualization,
          collapsed_categories=self._config.collapsed_categories,
          out_video_path=video_output_path,
          summary_logger=self._summary_logger,
      )

      self._run_frame_loop(image_paths, tracker, visualizer, frame_output_dir)

      visualizer.close_video()
      flush_memory()

      track_predictions = tracker.classify_all_tracks(
          self._classifier,
          self._config.models.dinov3.classification_batch_size,
      )
      track_summary = visualizer.save_track_grids(
          track_predictions, tracker.resolve_track_label, grid_output_dir
      )
      if self._config.tracking.enable:
        visualizer.print_summary(
            track_summary, input_subfolder, self._config.classes
        )
      else:
        logger.info("Tracking disabled: skipping per-subfolder summary.")

      track_rows = gcs_ops.build_track_rows(track_summary, subfolder_name)

      logger.info("Finished subfolder: %s", subfolder_name)
      return track_rows
    finally:
      detach_file_handler(file_handler)

  def _run_frame_loop(
      self,
      image_paths: list[str],
      tracker: Any,
      visualizer: Any,
      frame_output_dir: str,
  ) -> None:
    """Iterates over the image frames, running detection, tracking, and visualization.

    Args:
        image_paths: Sorted list of image file paths to process.
        tracker: Per-subfolder TrackManager (already instantiated).
        visualizer: Per-subfolder PipelineVisualizer (already instantiated).
        frame_output_dir: Directory where annotated frames are saved.
    """
    progress_bar = tqdm.tqdm(image_paths, desc="Tracking")

    for frame_index, image_path in enumerate(progress_bar):
      resized_image = load_and_resize_image(
          image_path, self._config.preprocessing.max_short_side
      )

      state = self._detector.detect(resized_image)
      detections = self._detector.to_supervision_detections(state)

      detections, unassigned_scores = tracker.update_and_extract_crops(
          detections, state, resized_image, os.path.basename(image_path)
      )

      if unassigned_scores:
        formatted_scores = ", ".join(
            f"{score:.3f}" for score in unassigned_scores
        )
        progress_bar.write(
            f"  frame {frame_index:06d} unassigned scores: {formatted_scores}"
        )

      frame_output_path = os.path.join(
          frame_output_dir, os.path.basename(image_path)
      )
      visualizer.annotate_and_write_frame(
          resized_image, detections, frame_output_path
      )

      del resized_image, state, detections
      if frame_index % _MEMORY_FLUSH_FRAME_INTERVAL == 0:
        flush_memory()

  def _collect_image_paths(self, directory: str) -> list[str]:
    """Collects and sorts image file paths matching the configured extensions.

    Args:
        directory: Directory to scan for image files (non-recursive).

    Returns:
        Sorted list of image file paths.
    """
    image_paths = []
    for extension_pattern in self._config.models.rfdetr.image_file_extensions:
      image_paths.extend(glob.glob(os.path.join(directory, extension_pattern)))
    return sorted(image_paths)

  def _discover_subfolders(self, root_directory: str) -> list[str]:
    """Returns sorted absolute paths of immediate (direct child) subfolders.

    Args:
        root_directory: Directory whose direct child folders should be returned.

    Returns:
        Sorted list of absolute subfolder paths.
    """
    logger = get_logger()
    if not os.path.isdir(root_directory):
      logger.error("Input directory does not exist: %s", root_directory)
      return []

    entries = os.listdir(root_directory)
    subfolders = [
        os.path.join(root_directory, entry)
        for entry in entries
        if os.path.isdir(os.path.join(root_directory, entry))
    ]
    return sorted(subfolders)


if __name__ == "__main__":
  pipeline = PETBottlePipeline(config_path="config.yaml")
  pipeline.run()
