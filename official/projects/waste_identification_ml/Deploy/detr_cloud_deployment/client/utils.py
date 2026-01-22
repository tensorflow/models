# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for the pipeline."""

import datetime
import logging
import os
import subprocess
import sys
import time

import natsort
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import supervision as sv


sys.path.append(
    'models/official/projects/waste_identification_ml/model_inference/'
)
import color_and_property_extractor  # pylint: disable=g-bad-import-order, g-import-not-at-top

_DETECTION_COLOR_PALETTE = sv.ColorPalette.from_hex([
    '#ffff00',
    '#ff9b00',
    '#ff66ff',
    '#3399ff',
    '#ff66b2',
    '#ff8080',
    '#b266ff',
    '#9999ff',
    '#66ffff',
    '#33ff99',
    '#66ff66',
    '#99ff00',
])


def _create_log_file(name: str, logs_folder_path: str) -> logging.Logger:
  """Creates a logger and a log file given the name of the video.

  Args:
    name: The name of the video.
    logs_folder_path: Path to the directory where logs should be saved.

  Returns:
    logging.Logger: Logger object configured to write logs to the file.
  """
  log_file_path = os.path.join(logs_folder_path, f'{name}.log')
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  file_handler = logging.FileHandler(log_file_path)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  return logger


def setup_logger_and_directories(input_dir):
  """Sets up directories and a logger for the pipeline.

  This function copies the input directory from GCP, creates a prediction
  folder, and initializes a logger for the current run.

  Args:
    input_dir: The path to the input directory on GCP.

  Returns:
    A tuple containing:
      - input_directory: The local path of the copied input directory.
      - prediction_folder: The path to the created prediction folder.
      - logger: The configured logging.Logger object.
  """

  input_directory = (input_dir).rstrip('/\\')
  command = f'gsutil -m cp -r {input_directory} .'
  subprocess.run(command, shell=True, check=True)
  prediction_folder = os.path.basename(input_directory) + '_prediction'
  os.makedirs(prediction_folder, exist_ok=True)
  log_name = os.path.basename(input_dir)
  log_folder = os.path.join(os.getcwd(), 'logs')
  os.makedirs(log_folder, exist_ok=True)
  logger = _create_log_file(log_name, log_folder)
  return input_directory, prediction_folder, logger


def get_class_id_to_class_name_mapping(label_csv):
  labels_path = os.path.join(os.getcwd(), label_csv)
  labels_df = pd.read_csv(labels_path)
  class_id_to_class_name_mapper = labels_df.set_index('id').to_dict()['names']
  return class_id_to_class_name_mapper


def get_image_capture_time(image_path):
  """Retrieves the creation time of an image, trying multiple methods.

  Args:
    image_path: The path to the image file.

  Returns:
    A string representing the creation time in the format "%Y-%m-%d %H:%M:%S" if
    found, otherwise returns "Creation time not found".
  """
  try:
    filename_str = os.path.basename(image_path)
    parts = filename_str.split('_')
    date_part = parts[-2]
    time_part = parts[-1].split('.')[0]

    # Format Date: YYYY-MM-DD
    formatted_date = f'{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}'

    # Format Time: HH:MM:SS.ms (Taking the last 2 digits as the decimal)
    formatted_time = (
        f'{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}.{time_part[6:]}'
    )

    return datetime.datetime.strptime(
        f'{formatted_date} {formatted_time}', '%Y-%m-%d %H:%M:%S.%f'
    ).strftime('%Y-%m-%d %H:%M:%S')

  except (IndexError, ValueError):
    try:

      # 1. Try EXIF data (if available)
      image = Image.open(image_path)
      exif_data = image.getexif()
      if exif_data:
        datetime_tag_id = 36867  # Tag ID for "DateTimeOriginal"
        datetime_str = exif_data.get(datetime_tag_id)
        if datetime_str:
          return datetime.datetime.strptime(
              datetime_str, '%Y:%m:%d %H:%M:%S'
          ).strftime('%Y-%m-%d %H:%M:%S')

      # 2. Try file modification time (less accurate, but better than nothing)
      file_modified_time = os.path.getmtime(image_path)
      return datetime.datetime.fromtimestamp(file_modified_time).strftime(
          '%Y-%m-%d %H:%M:%S'
      )
    except FileNotFoundError:
      return 'Image not found'
    except PIL.UnidentifiedImageError as e:
      return f'Error: {e}'
    except (OSError, PIL.ImageError) as e:
      return f'Error processing image or file: {e}'


def files_paths(folder_path):
  """List the full paths of image files in a folder and sort them.

  Args:
    folder_path: The path of the folder to list the image files from.

  Returns:
    A list of full paths of the image files in the folder, sorted in ascending
    order.
  """
  img_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
  image_files_full_path = []
  for entry in os.scandir(folder_path):
    if entry.is_file() and entry.name.lower().endswith(img_extensions):
      image_files_full_path.append(entry.path)

  # Sort the list of files by name
  image_files_full_path = natsort.natsorted(image_files_full_path)
  return image_files_full_path


def _crop_objects_from_masks(
    image: np.ndarray, masks: np.ndarray
) -> list[np.ndarray]:
  """Crops objects from an image based on provided masks.

  Args:
    image: The image as a numpy array.
    masks: A numpy array of binary masks, where each slice corresponds to an
      object.

  Returns:
    A list of numpy arrays, where each array is a cropped object.
  """
  cropped_objects = []
  for m in masks.astype(int):
    cropped_object = np.where(np.expand_dims(m, -1), image, 0)
    cropped_objects.append(cropped_object)
  return cropped_objects


def extract_color_names(results, image_for_saving):
  """Extracts generic color names from detected objects in an image.

  This function uses the provided masks in `results` to crop each detected
  object from `image_for_saving`. It then finds the dominant color for each
  cropped object and returns a list of generic color names.

  Args:
    results: A dictionary containing detection results, including 'masks' (numpy
      array of masks).
    image_for_saving: The image as a numpy array from which objects are cropped.

  Returns:
    A list of strings, where each string is a generic color name
    corresponding to a detected object.
  """
  # Crop objects from an image using masks for color detection.
  cropped_objects = _crop_objects_from_masks(
      image_for_saving, results['masks']
  )
  # Perform color detection using clustering approach.
  dominant_colors = [
      *map(color_and_property_extractor.find_dominant_color, cropped_objects)
  ]
  generic_color_names = color_and_property_extractor.get_generic_color_name(
      dominant_colors
  )
  return generic_color_names


def draw_detections_and_save_image(img, results, save_path):
  """Used for plotting the annotations on the image.

  Args:
    img: The PIL Image object to plot annotations on.
    results: A dictionary containing detection results.
    save_path: The file path to save the annotated image.

  Returns:
    A PIL Image object with the annotations plotted.
  """

  detection = sv.Detections(
      xyxy=results['xyxy'],
      mask=results['masks'],
      confidence=results['confidence'],
      class_id=results['labels'],
      data={'class_names': results['class_names']},
  )

  text_scale = sv.calculate_optimal_text_scale(resolution_wh=img.size)
  thickness = sv.calculate_optimal_line_thickness(resolution_wh=img.size)
  color = _DETECTION_COLOR_PALETTE
  detections_labels = [
      f'{class_name} : {probability:.2f}'
      for class_name, probability in zip(
          detection.data['class_names'], detection.confidence
      )
  ]
  detections_image = sv.MaskAnnotator(opacity=0.4).annotate(
      scene=img.copy(), detections=detection
  )
  detections_image = sv.BoxAnnotator(color=color, thickness=thickness).annotate(
      detections_image, detection
  )
  detections_image = sv.LabelAnnotator(
      color=color, text_color=sv.Color.BLACK, text_scale=text_scale
  ).annotate(detections_image, detection, detections_labels)
  image_with_detections = Image.new(
      'RGB', (img.width + detections_image.width, img.height)
  )
  image_with_detections.paste(img, (0, 0))
  image_with_detections.paste(detections_image, (img.width, 0))
  image_with_detections.save(save_path)


def shutdown_vm():
  """Shuts down the system."""
  time.sleep(60)
  print('Attempting to shut down the VM instance...')
  try:
    command = ['sudo', 'poweroff']
    subprocess.run(command, check=True)
  except subprocess.CalledProcessError as e:
    print(f'Failed to shut down: {e}')
