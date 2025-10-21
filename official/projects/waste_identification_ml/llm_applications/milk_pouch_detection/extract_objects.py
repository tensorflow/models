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

"""Detects, segments, and saves objects from images in a directory.

This script initializes a computer vision pipeline to process images, identify
objects based on a text prompt, and save each detected object as a separate
cropped image to a temporary directory.
"""

import glob
import json
import os
from typing import List, Tuple
import warnings

from absl import app
from absl import flags
import batched_io
import models
import models_utils
import natsort
import numpy as np
import torch
import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


GROUNDING_DINO_WEIGHTS = (
    "milk_pouch_project/grounding_dino_model/groundingdino_swint_ogc.pth"
)
GROUNDING_DINO_CONFIG = (
    "milk_pouch_project/grounding_dino_model/GroundingDINO_SwinT_OGC.py"
)
SAM2_WEIGHTS = "milk_pouch_project/sam2_model/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
TEXT_PROMPT = "packets"
INPUT_DIR = "input_images"
CLASSIFICATION_DIR = "objects_for_classification"
COCO_OUTPUT_PATH = "coco_output.json"

# Filter generated masks that are less than or equal to this
# percentage of the overall image area
MASK_FILTER_THRESHOLD_PERCENT = 1.0

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "category_name",
    None,
    "Name of the category. If provided, creates a COCO JSON file.",
)


def filter_masks_by_area(
    masks: List[np.ndarray],
    boxes: List[np.ndarray],
    image_area: int,
    min_percent: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  """Filter masks and boxes by minimum area threshold.

  Args:
    masks: Binary masks.
    boxes: Bounding boxes corresponding to masks.
    image_area: Total area of the source image.
    min_percent: Minimum mask area as percentage of image area.

  Returns:
    Filtered masks and boxes.
  """
  filtered = [
      (mask, box)
      for mask, box in zip(masks, boxes)
      if (np.sum(mask.astype(np.uint8)) / image_area) * 100 > min_percent
  ]

  if not filtered:
    return [], []

  valid_masks, valid_boxes = zip(*filtered)
  return list(valid_masks), list(valid_boxes)


def main(_) -> None:
  """Runs the main object detection and extraction pipeline."""
  if not os.path.isdir(INPUT_DIR):
    raise ValueError(f"Input directory not found at '{INPUT_DIR}'")

  # Check if COCO output should be created
  create_coco = FLAGS.category_name is not None

  print("Initializing image extraction and classification...")
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
  os.makedirs(CLASSIFICATION_DIR, exist_ok=True)

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
  all_files = glob.glob(os.path.join(INPUT_DIR, "*"))
  image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
  files = [f for f in all_files if f.lower().endswith(image_extensions)]
  files = natsort.natsorted(files)

  writer = batched_io.BatchedMaskWriter(
      CLASSIFICATION_DIR, max_workers=FLAGS.io_workers
  )

  try:
    for image_id_counter, file_path in tqdm.tqdm(enumerate(files)):
      try:
        with torch.no_grad():
          results = pipeline.detect_and_segment(file_path, TEXT_PROMPT)
        if not results:
          print("No objects detected")
          continue
      except (RuntimeError, ValueError) as e:
        print(
            "An unexpected error occurred with"
            f" {os.path.basename(file_path)}: {e}"
        )
        continue

      image = results["image"]
      h, w = image.shape[:2]
      image_area = h * w

      valid_masks, valid_boxes = filter_masks_by_area(
          results["masks"],
          results["boxes"],
          image_area,
          min_percent=MASK_FILTER_THRESHOLD_PERCENT,
      )

      # Add image info to COCO output only if create_coco is True
      if create_coco:
        image_info = {
            "id": image_id_counter,
            "file_name": os.path.basename(file_path),
            "width": w,
            "height": h,
        }
        coco_output["images"].append(image_info)

      writer.add_batch(
          image,
          valid_masks,
          valid_boxes,
          file_path,
      )

      if create_coco:
        for _, (box, mask) in enumerate(zip(valid_boxes, valid_masks)):
          try:
            # Get the polygon points of masks
            segmentation = models_utils.extract_largest_contour_segmentation(
                mask
            )
            bbox_width, bbox_height, area = models_utils.get_bbox_details(box)

            # Annotation key format in COCO JSON
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
                "segmentation": segmentation,
            }
            coco_output["annotations"].append(annotation_info)
            annotation_id_counter += 1
          except (ValueError, SystemError) as e:
            print(f"[ERROR] Failed to create annotation: {e}")
            continue
  finally:
    # Ensure all I/O operations complete
    if writer:
      writer.__exit__(None, None, None)

  # Save COCO JSON file only if create_coco is True
  if create_coco:
    with open(os.path.join(INPUT_DIR, COCO_OUTPUT_PATH), "w") as f:
      json.dump(coco_output, f, indent=4)
    print(f"\n✅ Processing complete. COCO JSON saved to '{COCO_OUTPUT_PATH}'.")

  print(f"✅ Cropped images saved to '{CLASSIFICATION_DIR}'.")


if __name__ == "__main__":
  app.run(main)
