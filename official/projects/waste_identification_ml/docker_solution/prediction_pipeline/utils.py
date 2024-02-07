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

"""Utils for docker solution pipeline.

These functions provide basic yet essential operations that facilitate various
steps in the data processing and management workflow. The script is intended to
be a shared resource across different modules or scripts within a project,
offering common functionalities.

The script includes key function like reading images, creating log files,
creating folders and changing data types.
"""

import logging
import os
import cv2
import numpy as np


HEIGHT, WIDTH = 512, 1024


def read_image(path: str) -> np.ndarray:
  """Reads an image.

  This function uses OpenCV to read an image from the given path. The read image
  is then resized
  to the dimensions specified by the WIDTH and HEIGHT constants. It uses
  INTER_AREA interpolation,
  which is generally best for shrinking an image.

  Args:
    path: The file path of the image to be read.

  Returns:
    The resized image as a numpy array
  """
  image = cv2.imread(path)
  image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
  return image


def create_log_file(video_name: str, logs_folder_path: str) -> logging.Logger:
  """Creates a logger and a log file given the name of the video.

  Args:
    video_name: The name of the video.
    logs_folder_path: Path to the directory where logs should be saved.

  Returns:
    logging.Logger: Logger object configured to write logs to the file.
  """
  log_file_path = os.path.join(logs_folder_path, f"{video_name}.log")
  logger = logging.getLogger(video_name)
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(log_file_path)
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  return logger


def create_folders_from_video_name(video_name: str) -> tuple[str, str, str]:
  """Creates three folders based on the given video name.

  One for frames, one for predictions, and one for masks.
  Each folder's name is derived from the video name, excluding its file
  extension.

  Args:
    video_name: The name of the video file, including its extension.

  Returns:
    A tuple containing the names of the created folders:(frames_folder_name,
    prediction_folder_name, masks_folder_name).
  """
  base_folder_name, _ = os.path.splitext(video_name)
  folder_suffixes = ["", "_frames", "_masks"]
  folder_names = []

  for suffix in folder_suffixes:
    folder_name = base_folder_name + suffix
    os.makedirs(folder_name, exist_ok=True)
    folder_names.append(folder_name)

  return tuple(folder_names)


def convert_and_change_dtype(
    data: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
  """Change the data types.

  Change the values from int64 to int32 type and float64 to float32 type.

  Args:
    data: A dictionary with values whose value data types need to be changed.

  Returns:
    The dictionary with changed data types.
    changed as necessary.
  """
  for key, value in data.items():
    value = np.array(value)

    if value.dtype == np.int64:
      value = value.astype(np.int32)
    elif value.dtype == np.float64:
      value = value.astype(np.float32)

    data[key] = value
  return data
