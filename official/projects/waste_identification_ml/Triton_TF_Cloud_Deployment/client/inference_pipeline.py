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

"""Pipeline to run the prediction on the images folder with Triton server."""

import os
import subprocess
import sys

from absl import app
from absl import flags
import big_query_ops
import cv2
import feature_extraction
import ffmpeg_ops
import mask_bbox_saver
import numpy as np
import object_tracking
import object_tracking_postprocessing
import pandas as pd
import triton_server_inference
import utils

sys.path.append(
    "models/official/projects/waste_identification_ml/model_inference/"
)
import color_and_property_extractor  # pylint: disable=g-bad-import-order, g-import-not-at-top

INPUT_DIRECTORY = flags.DEFINE_string(
    "input_directory", None, "The path to the directory containing images."
)

OUTPUT_DIRECTORY = flags.DEFINE_string(
    "output_directory", None, "The path to the directory to save the results."
)

HEIGHT = flags.DEFINE_integer(
    "height", None, "Height of an image required by the model"
)

WIDTH = flags.DEFINE_integer(
    "width", None, "Width of an image required by the model"
)

MODEL = flags.DEFINE_string("model", None, "Model name")

PREDICTION_THRESHOLD = flags.DEFINE_float(
    "score", None, "Threshold to filter the prediction results"
)

SEARCH_RANGE_X = flags.DEFINE_integer(
    "search_range_x",
    None,
    "Pixels upto which every object needs to be tracked along X axis.",
)

SEARCH_RANGE_Y = flags.DEFINE_integer(
    "search_range_y",
    None,
    "Pixels upto which every object needs to be tracked along Y axis.",
)

MEMORY = flags.DEFINE_integer(
    "memory", None, "Frames upto which every object needs to be tracked."
)

OVERWRITE = flags.DEFINE_boolean(
    "overwrite",
    False,
    "If True, delete the preexisting BigQuery table before creating a new one.",
)

PROJECT_ID = flags.DEFINE_string(
    "project_id", None, "Project ID mentioned in Google Cloud Project"
)

BQ_DATASET_ID = flags.DEFINE_string(
    "bq_dataset_id", "Circularnet_dataset", "Big query dataset ID"
)

BQ_TABLE_ID = flags.DEFINE_string(
    "bq_table_id", "Circularnet_table", "BigQuery Table ID for features data"
)

TRACKING_VISUALIZATION = flags.DEFINE_boolean(
    "tracking_visualization",
    False,
    "If True, visualize the tracking results.",
)

CROPPED_OBJECTS = flags.DEFINE_boolean(
    "cropped_objects",
    False,
    "If True, save cropped objects per category from the output.",
)

AREA_THRESHOLD = None
HEIGHT_TRACKING = 300
WIDTH_TRACKING = 300
CIRCLE_RADIUS = 7
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (255, 0, 0)


def main(_) -> None:
  # Check if the input and output directories are valid.
  if (
      not INPUT_DIRECTORY.value
      or not OUTPUT_DIRECTORY.value
      or not INPUT_DIRECTORY.value.startswith("gs://")
      or not OUTPUT_DIRECTORY.value.startswith("gs://")
  ):
    raise ValueError("Bucket path must be non-empty starting with 'gs://'")

  # Copy the images folder from GCP to the present directory.
  input_directory = (INPUT_DIRECTORY.value).rstrip("/\\")
  command = f"gsutil -m cp -r {input_directory} ."
  subprocess.run(command, shell=True, check=True)

  # Create a folder to store the predictions.
  prediction_folder = os.path.basename(input_directory) + "_prediction"
  os.makedirs(prediction_folder, exist_ok=True)

  # Create a log directory and a logger for logging.
  log_name = os.path.basename(INPUT_DIRECTORY.value)
  log_folder = os.path.join(os.getcwd(), "logs")
  os.makedirs(log_folder, exist_ok=True)
  logger = utils.create_log_file(log_name, log_folder)

  # Read the labels which the model is trained on.
  labels_path = os.path.join(os.getcwd(), "labels.csv")
  labels, category_index = utils.load_labels(labels_path)

  # Read files from a folder.
  files = utils.files_paths(os.path.basename(input_directory))

  tracking_images = {}
  features_set = []
  image_plot = None
  agg_features = None
  tracking_features = None

  for frame, image_path in enumerate(files, start=1):
    # Prepare an input for a Triton model server from an image.
    logger.info(f"Processing {os.path.basename(image_path)}")
    try:
      inputs, original_image, _ = (
          triton_server_inference.prepare_image(
              image_path, HEIGHT.value, WIDTH.value
          )
      )

      # Extract the creation time of an image.
      creation_time = ffmpeg_ops.get_image_creation_time(image_path)
      logger.info(
          f"Successfully read an image for {os.path.basename(image_path)}"
      )
    except (cv2.error, ValueError, TypeError) as e:
      logger.info("Failed to read an image.")
      logger.exception("Exception occurred:", e)
      continue

    try:
      model_name = MODEL.value
      result = triton_server_inference.infer(model_name, inputs)
      logger.info(
          f"Successfully got prediction for {os.path.basename(image_path)}"
      )
      logger.info(f"Total predictions:{result['num_detections'][0]}")
    except (KeyError, TypeError, RuntimeError, ValueError) as e:
      logger.info(
          f"Failed to get prediction for {os.path.basename(image_path)}"
      )
      logger.exception("Exception occurred:", e)
      continue

    try:
      # Take predictions only above the threshold.
      if result["num_detections"][0]:
        scores = result["detection_scores"][0]
        filtered_indices = scores > PREDICTION_THRESHOLD.value

        if any(filtered_indices):
          result = utils.filter_detections(result, filtered_indices)
          logger.info(
              "Total predictions after"
              f" thresholding:{result['num_detections'][0]}"
          )
      else:
        logger.info("Zero predictions after threshold.")
        continue
    except (KeyError, IndexError, TypeError, ValueError) as e:
      logger.info("Failed to filter out predictions.")
      logger.exception("Exception occured:", e)

    try:
      # Convert bbox coordinates into normalized coordinates.
      if result["num_detections"][0]:
        result["normalized_boxes"] = result["detection_boxes"].copy()
        result["normalized_boxes"][:, :, [0, 2]] /= HEIGHT.value
        result["normalized_boxes"][:, :, [1, 3]] /= WIDTH.value
        result["detection_boxes"] = (
            result["detection_boxes"].round().astype(int)
        )

        # Adjust the image size to ensure both dimensions are at least 1024
        # for saving images with bbox and masks.
        height_plot, width_plot = utils.adjust_image_size(
            original_image.shape[0], original_image.shape[1], 1024
        )

        # Resize the original image to overlay bbox and masks on it.
        image_plot = cv2.resize(
            original_image,
            (width_plot, height_plot),
            interpolation=cv2.INTER_AREA,
        )

        # Reframe the masks according to the new image size.
        result["detection_masks_reframed"] = utils.reframe_masks(
            result, "normalized_boxes", height_plot, width_plot
        )

        # Filter the prediction results and remove the overlapping masks.
        unique_indices = utils.filter_masks(
            result["detection_masks_reframed"],
            iou_threshold=0.08,
            area_threshold=AREA_THRESHOLD,
        )
        result = utils.filter_detections(result, unique_indices)
        logger.info(
            f"Total predictions after processing: {result['num_detections'][0]}"
        )
      else:
        logger.info("Zero predictions after processing.")
        continue
    except (KeyError, IndexError, TypeError, ValueError) as e:
      logger.info("Issue in post processing predictions results.")
      logger.exception("Exception occured:", e)

    try:
      if result["num_detections"][0]:
        result["detection_classes_names"] = np.array(
            [[str(labels[i - 1]) for i in result["detection_classes"][0]]]
        )

        # Save the prediction results as an image file with bbx and masks.
        mask_bbox_saver.save_bbox_masks_labels(
            result=result,
            image=image_plot,
            file_name=os.path.basename(image_path),
            folder=prediction_folder,
            category_index=category_index,
            threshold=PREDICTION_THRESHOLD.value,
        )
        logger.info("Visualization saved.")
    except (KeyError, IndexError, TypeError, ValueError) as e:
      logger.info("Issue in saving visualization of results.")
      logger.exception("Exception occured:", e)

    try:
      # Resize an image for object tracking..
      tracking_image = cv2.resize(
          original_image,
          (WIDTH_TRACKING, HEIGHT_TRACKING),
          interpolation=cv2.INTER_AREA,
      )
      tracking_images[os.path.basename(image_path)] = tracking_image

      # Reducing mask sizes in order to keep the memory required for object
      # tracking under a threshold.
      result["detection_masks_tracking"] = np.array([
          cv2.resize(
              i,
              (WIDTH_TRACKING, HEIGHT_TRACKING),
              interpolation=cv2.INTER_NEAREST,
          )
          for i in result["detection_masks_reframed"]
      ])

      # Crop objects from an image using masks for color detection.
      cropped_objects = [
          np.where(np.expand_dims(i, -1), image_plot, 0)
          for i in result["detection_masks_reframed"]
      ]

      # Perform color detection using clustering approach.
      dominant_colors = [
          *map(
              color_and_property_extractor.find_dominant_color, cropped_objects
          )
      ]
      generic_color_names = color_and_property_extractor.get_generic_color_name(
          dominant_colors
      )

      # Extract features.
      features = feature_extraction.extract_properties(
          tracking_image, result, "detection_masks_tracking"
      )
      features["source_name"] = os.path.basename(os.path.dirname(image_path))
      features["image_name"] = os.path.basename(image_path)
      features["creation_time"] = creation_time
      features["frame"] = frame
      features["detection_scores"] = result["detection_scores"][0]
      features["detection_classes"] = result["detection_classes"][0]
      features["detection_classes_names"] = result["detection_classes_names"][0]
      features["color"] = generic_color_names
      features_set.append(features)
      logger.info("Features extracted.\n")
    except (KeyError, IndexError, TypeError, ValueError):
      logger.info("Failed to extract properties.")

  try:
    if features_set:
      features_df = pd.concat(features_set, ignore_index=True)

      # Apply object tracking to the features.
      tracking_features = object_tracking.apply_tracking(
          features_df,
          search_range_x=SEARCH_RANGE_X.value,
          search_range_y=SEARCH_RANGE_Y.value,
          memory=MEMORY.value,
      )

      # Process the tracking results to remove errors.
      agg_features = object_tracking_postprocessing.process_tracking_result(
          tracking_features
      )
      counts = agg_features.groupby("detection_classes_names").size()
      counts.to_frame().to_csv(os.path.join(os.getcwd(), "count.csv"))
      logger.info("Object tracking applied.")
  except (KeyError, IndexError, TypeError, ValueError):
    logger.info("Failed to apply object tracking.")

  try:
    if TRACKING_VISUALIZATION.value:
      # Create a folder to save the tracking visualization.
      tracking_folder = os.path.basename(input_directory) + "_tracking"
      os.makedirs(tracking_folder, exist_ok=True)

      # Save the tracking results as an image files.
      output_folder = mask_bbox_saver.visualize_tracking_results(
          tracking_features=tracking_features,
          tracking_images=tracking_images,
          tracking_folder=tracking_folder,
      )
      logger.info(f"Tracking visualization saved to {output_folder}.")

      # Move the tracking visualization to the output directory.
      commands = [
          f"gsutil -m cp -r {output_folder} {OUTPUT_DIRECTORY.value}",
          f"rm -r {output_folder}",
      ]
      combined_command_1 = " && ".join(commands)
      subprocess.run(combined_command_1, shell=True, check=True)
      logger.info("Tracking visualization saved.")
  except (KeyError, IndexError, TypeError, ValueError):
    logger.info("Failed to visualize tracking results.")

  try:
    if CROPPED_OBJECTS.value:
      cropped_obj_folder = mask_bbox_saver.save_cropped_objects(
          agg_features=agg_features,
          input_directory=input_directory,
          height_tracking=HEIGHT_TRACKING,
          width_tracking=WIDTH_TRACKING,
          resize_bbox=utils.resize_bbox,
      )
      logger.info("Cropped objects saved in %s", cropped_obj_folder)

      # Move the cropped objects to the output directory.
      commands = [
          f"gsutil -m cp -r {cropped_obj_folder} {OUTPUT_DIRECTORY.value}",
          f"rm -r {cropped_obj_folder}",
      ]

      combined_command_2 = " && ".join(commands)
      subprocess.run(combined_command_2, shell=True, check=True)
      logger.info("Cropped objects saved.")
  except (KeyError, IndexError, TypeError, ValueError):
    logger.info("Issue in cropping objects")
    logger.info("Failed to crop objects.")
    logger.exception("Exception occured:", e)

  if isinstance(agg_features, pd.DataFrame) and not agg_features.empty:
    try:
      # Create a big query table to store the aggregated features data.
      big_query_ops.create_table(
          PROJECT_ID.value,
          BQ_DATASET_ID.value,
          BQ_TABLE_ID.value,
          overwrite=OVERWRITE.value,
      )
      logger.info("Successfully created table.")
    except (KeyError, IndexError, TypeError, ValueError):
      logger.info("Issue in creation of table")
      return

    try:
      # Ingest the aggregated features data into the big query table.
      big_query_ops.ingest_data(
          agg_features, PROJECT_ID.value, BQ_DATASET_ID.value, BQ_TABLE_ID.value
      )
      logger.info("Data ingested successfully.")
    except (KeyError, IndexError, TypeError, ValueError):
      logger.info("Issue in data ingestion.")
      return

    try:
      # Move the folders to the destination bucket.
      commands = [
          (
              "gsutil -m cp -r"
              f" {os.path.basename(input_directory)} {OUTPUT_DIRECTORY.value}"
          ),
          f"rm -r {os.path.basename(input_directory)}",
          f"gsutil -m cp -r {prediction_folder} {OUTPUT_DIRECTORY.value}",
          f"rm -r {prediction_folder}",
      ]

      combined_command_3 = " && ".join(commands)
      subprocess.run(combined_command_3, shell=True, check=True)
      logger.info("Successfully moved to destination bucket")
    except (KeyError, IndexError, TypeError, ValueError):
      logger.info("Issue in moving folders to destination bucket")
  else:
    logger.info("No features to ingest.")


if __name__ == "__main__":
  app.run(main)
