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

"""Utility functions for the pet grading cloud deployment."""

import collections
import datetime
import gc
import itertools
import logging
import math
import os
import pathlib
import re
import subprocess
import matplotlib.pyplot as plt
import natsort
import numpy as np


def _create_log_file(name: str, logs_folder_path: str) -> logging.Logger:
  """Creates a logger and a log file given the name of the video.

  Args:
    name: The name of the video.
    logs_folder_path: Path to the directory where logs should be saved.

  Returns:
    logging.Logger: Logger object configured to write logs to the file.
  """
  log_file_path = os.path.join(logs_folder_path, f"{name}.log")
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(log_file_path)
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  return logger


def parse_imgage_names_to_datetime(filename):
  """Parses an image filename to extract a datetime object.

  The filename is expected to be in the format "img_YYYYMMDD_HHMMSSms.*".

  Args:
    filename: The path to the image file.

  Returns:
    A datetime.datetime object parsed from the filename.
  """
  stem = pathlib.Path(filename).stem
  m = re.match(r"img_(\d{8})_(\d{6})(\d*)", stem)
  dt_str = f"{m.group(1)}_{m.group(2)}{m.group(3)}"
  return datetime.datetime.strptime(dt_str, "%Y%m%d_%H%M%S%f")


def files_paths(folder_path):
  """List the full paths of image files in a folder and sort them.

  Args:
    folder_path: The path of the folder to list the image files from.

  Returns:
    A list of full paths of the image files in the folder, sorted in ascending
    order.
  """
  img_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
  image_files_full_path = []
  for entry in os.scandir(folder_path):
    if entry.is_file() and entry.name.lower().endswith(img_extensions):
      image_files_full_path.append(entry.path)

  try:
    image_files_full_path = sorted(
        image_files_full_path, key=parse_imgage_names_to_datetime
    )
  except AttributeError:
    print("Failed to parse image names to datetime, using natsort instead.")
    image_files_full_path = natsort.natsorted(image_files_full_path)
  return image_files_full_path


def setup_logger_and_directories(input_dir):
  """Sets up directories and a logger for the pipeline.

  Args:
    input_dir: The path to the input directory on GCP.

  Returns:
    A tuple containing:
      - input_directory: The local path of the copied input directory.
      - prediction_folder: The path to the created prediction folder.
      - logger: The configured logging.Logger object.
  """

  input_directory = (input_dir).rstrip("/\\")
  command = f"gcloud storage cp -r {input_directory} ."
  subprocess.run(command, shell=True, check=True)
  prediction_folder = os.path.basename(input_directory) + "_prediction"
  os.makedirs(prediction_folder, exist_ok=True)
  log_name = os.path.basename(input_dir)
  log_folder = os.path.join(os.getcwd(), "logs")
  os.makedirs(log_folder, exist_ok=True)
  logger = _create_log_file(log_name, log_folder)
  return input_directory, prediction_folder, logger


def get_class_with_majority_vote(list_of_predictions, min_probability=50):
  """Determines consensus predictions for each tracker using majority voting.

  Args:
    list_of_predictions: A list of prediction dictionaries, where each must
      contain 'tracker_id', 'predicted_class', and 'predicted_probability'.
    min_probability: The minimum required probability threshold for a tracker to
      be considered valid. Default is 50.

  Returns:
    A list of prediction dictionaries for the trackers that met the minimum
    probability threshold, with each dictionary updated with its tracker's
    'best_class' and 'best_probability'.
  """
  tracker_probs = collections.defaultdict(lambda: collections.defaultdict(list))
  for p in list_of_predictions:
    tracker_probs[p["tracker_id"]][p["predicted_class"]].append(
        p["predicted_probability"]
    )

  tracker_id_to_best_class_dict = {}
  for tid, class_probs in tracker_probs.items():
    vote_counts = {c: len(probs) for c, probs in class_probs.items()}
    max_votes = max(vote_counts.values())
    max_voted_class = [c for c, v in vote_counts.items() if v == max_votes]

    if len(max_voted_class) == 1:
      best_class = max_voted_class[0]
    else:
      # In case of tiebreak, choose by average probability
      best_class = max(
          max_voted_class,
          key=lambda c, class_probs=class_probs: max(class_probs[c]),
      )

    best_probability = max(class_probs[best_class])
    tracker_id_to_best_class_dict[tid] = {
        "best_class": best_class,
        "best_probability": best_probability,
    }

  valid_trackers = {
      tid: info
      for tid, info in tracker_id_to_best_class_dict.items()
      if info["best_probability"] >= min_probability
  }

  filtered = [
      p for p in list_of_predictions if p["tracker_id"] in valid_trackers
  ]
  for p in filtered:
    p.update(valid_trackers[p["tracker_id"]])

  return filtered


def save_output_image_grids(
    all_predictions, output_dr, columns_per_row=5, thumbnail_size=3
):
  """Saves a grid of thumbnail images with prediction labels for each frame.

  Args:
    all_predictions: A list of prediction dictionaries, where each contains
      'frame_name', 'tracker_id', 'best_class', 'best_probability', and 'crop'
      (the image array).
    output_dr: The output directory path where the saved image files are stored.
    columns_per_row: The maximum number of image columns in a grid row. Default
      is 5.
    thumbnail_size: The figure height/width factor (in inches) for each cell in
      the grid. Default is 3.
  """
  try:

    keyfunc = lambda p: p["frame_name"]
    sorted_preds = sorted(all_predictions, key=keyfunc)

    for frame_name, group in itertools.groupby(sorted_preds, key=keyfunc):
      rows = list(group)
      n = len(rows)

      ncols = min(n, columns_per_row)
      nrows = math.ceil(n / ncols)

      fig, axes = plt.subplots(
          nrows, ncols, figsize=(ncols * thumbnail_size, nrows * thumbnail_size)
      )
      fig.suptitle(frame_name, fontsize=9, fontweight="bold")

      if n == 1:
        axes = np.array([axes])
      axes = axes.flatten()

      for i, p in enumerate(rows):
        title = (
            f"#{p['tracker_id']} {p['best_class']}"
            f"\n{p['best_probability']:.3f}%"
        )
        axes[i].imshow(p["crop"])
        axes[i].set_title(title, fontsize=7)
        axes[i].axis("off")

      for j in range(n, len(axes)):
        axes[j].axis("off")

      plt.tight_layout()
      fig.savefig(f"{output_dr}/{frame_name}")
      plt.close(fig)
  except (KeyError, IndexError, TypeError, ValueError) as e:
    print(f"Issue in saving visualization of results, due to error : {e}")
  finally:
    gc.collect()
