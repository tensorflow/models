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

"""Detects, segments, and saves objects from images in a directory.

This script initializes a computer vision pipeline to process images, identify
objects based on a text prompt, and save each detected object as a separate
cropped image to a temporary directory.
"""

import glob
import json
import os
import warnings

from absl import app
from absl import flags
import models
import models_utils
import natsort
import numpy as np
import torch
import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Path to the pre-trained weights for the Grounding DINO model.
GROUNDING_DINO_WEIGHTS = (
    "milk_pouch_project/grounding_dino_model/groundingdino_swint_ogc.pth"
)

# Path to the configuration file for the Grounding DINO model.
GROUNDING_DINO_CONFIG = (
    "milk_pouch_project/grounding_dino_model/GroundingDINO_SwinT_OGC.py"
)

# Path to the pre-trained weights for the SAM2 model.
SAM2_WEIGHTS = "milk_pouch_project/sam2_model/sam2.1_hiera_large.pt"

# Path to the configuration file for the SAM2 model.
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Text prompt to use for object detection.
TEXT_PROMPT = "packets"

# Name of the temporary directory to store cropped object images.
TEMP_DIR = "tempdir"

# Path to save the output COCO dataset file.
COCO_OUTPUT_PATH = "coco_output.json"

# Minimum mask area as percentage of image.
MIN_MASK_AREA_PERCENT = 1.0

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_dir",
    None,
    "Directory containing image frames to process.",
    required=True,
)
flags.DEFINE_string(
    "category_name",
    None,
    "Name of the category. If provided, creates a COCO JSON file.",
)


def main(_) -> None:
  """Runs the main object detection and extraction pipeline."""
  if not os.path.isdir(FLAGS.input_dir):
    raise ValueError(f"Input directory not found at '{FLAGS.input_dir}'")

  # Check if COCO output should be created
  create_coco = FLAGS.category_name is not None

  print("Initializing Vision and Llm Pipeline...")
  try:
    pipeline = models.ObjectDetectionSegmentation(
        dino_config_path=GROUNDING_DINO_CONFIG,
        dino_weights_path=GROUNDING_DINO_WEIGHTS,
        sam_config_path=SAM2_CONFIG,
        sam_checkpoint_path=SAM2_WEIGHTS,
    )
  except FileNotFoundError as e:
    print(
        f"\n⚠️ Error: Could not find model files: {e}. "
        "Please check the paths in the script."
    )
    return
  print("✅ Pipeline ready.")

  # Create output directory
  output_dir = os.path.join(FLAGS.input_dir, TEMP_DIR)
  os.makedirs(output_dir, exist_ok=True)

  # Initialize coco json file format only if category_name is provided
  coco_output = None
  annotation_id_counter = 0
  if create_coco:
    coco_output = models_utils.initialize_coco_output(FLAGS.category_name)
    print(
        f"COCO JSON output will be created for category: {FLAGS.category_name}"
    )
  else:
    print("No category name provided. Skipping COCO JSON creation.")

  # Get all image files.
  all_files = glob.glob(os.path.join(FLAGS.input_dir, "*"))
  image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
  files = [f for f in all_files if f.lower().endswith(image_extensions)]
  files = natsort.natsorted(files)

  print(f"Processing {len(files)} images...")
  for image_id_counter, file_path in tqdm.tqdm(enumerate(files)):
    try:
      # Use no_grad context manager for inference to save memory.
      with torch.no_grad():
        results = pipeline.detect_and_segment(file_path, TEXT_PROMPT)

      if not results:
        print("No objects detected")
        continue  # No objects found or an issue occurred.
    except (RuntimeError, ValueError) as e:
      print(
          "An unexpected error occurred with"
          f" {os.path.basename(file_path)}: {e}"
      )
      continue

      # Uncomment to filter bigger overlapping boxes.
      # filtered_results = pipeline.filter_boxes_keep_smaller(
      #    results,
      #    iou_threshold=0.95
      # )

    image = results["image"]
    h, w = image.shape[:2]
    image_area = h * w

    # Add image info to COCO output only if create_coco is True
    if create_coco:
      image_info = {
          "id": image_id_counter,
          "file_name": os.path.basename(file_path),
          "width": w,
          "height": h,
      }
      coco_output["images"].append(image_info)

    for idx, (box, mask) in enumerate(
        zip(results["boxes"], results["masks"])
    ):
      mask_area = np.sum(mask.astype(np.uint8))
      if (mask_area / image_area) * 100 > 1:
        try:
          masked_object = models_utils.extract_masked_object(image, mask, box)
          models_utils.save_masked_object(
              masked_object, file_path, idx, TEMP_DIR
          )
        except (ValueError, SystemError, AttributeError, OSError) as e:
          print(
              f"[ERROR] Skipped saving image for {file_path} at index"
              f" {idx} due to: {e}"
          )
          continue

        # Add annotation info to COCO output only if create_coco is True
        if create_coco:
          # Get the polygon points of masks.
          segmentation = models_utils.extract_largest_contour_segmentation(mask)

          # coco bbox format
          bbox_width, bbox_height, area = models_utils.get_bbox_details(box)

          # annotation key format in coco json
          annotation_info = {
              "id": annotation_id_counter,
              "image_id": image_id_counter,
              "category_id": 1,
              "bbox": [
                  int(box[0]),
                  int(box[1]),
                  int(bbox_width),
                  int(bbox_height),
              ],
              "area": int(area),
              "iscrowd": 0,
              "segmentation": (
                  segmentation
              ),  # Optional: Add segmentation if you have it
          }
          coco_output["annotations"].append(annotation_info)
          annotation_id_counter += 1

  # Save COCO JSON file only if create_coco is True
  if create_coco:
    with open(os.path.join(FLAGS.input_dir, COCO_OUTPUT_PATH), "w") as f:
      json.dump(coco_output, f, indent=4)
    print(f"\n✅ Processing complete. COCO JSON saved to '{COCO_OUTPUT_PATH}'.")

  print(f"✅ Cropped images saved to '{TEMP_DIR}'.")


if __name__ == "__main__":
  app.run(main)
