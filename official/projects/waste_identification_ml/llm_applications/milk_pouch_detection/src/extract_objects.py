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
import os
from typing import List, Tuple
import warnings

from absl import app
from absl import flags
import natsort
import numpy as np
import torch
import tqdm

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import batched_io
from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import coco_annotation_writer
from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src.models import detection_segmentation


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


GROUNDING_DINO_WEIGHTS = "models/grounding_dino/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG = "models/grounding_dino/GroundingDINO_SwinT_OGC.py"
SAM2_WEIGHTS = "models/sam2/sam2.1_hiera_large.pt"
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
    pipeline = detection_segmentation.ObjectDetectionSegmentation(
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

  # Initialize COCO annotation writer only if category_name is provided
  coco_writer = None
  if create_coco:
    coco_writer = coco_annotation_writer.CocoAnnotationWriter(
        FLAGS.category_name
    )
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

  writer = batched_io.BatchedMaskWriter(CLASSIFICATION_DIR)

  try:
    for file_path in tqdm.tqdm(files):
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

      writer.add_batch(
          image,
          valid_masks,
          valid_boxes,
          file_path,
      )

      # Add image info to COCO output only if create_coco is True
      if create_coco and coco_writer:
        current_image_id = coco_writer.add_image(file_path, w, h)
        coco_writer.add_annotations(current_image_id, valid_boxes, valid_masks)

  finally:
    # Ensure all I/O operations complete
    if writer:
      writer.__exit__(None, None, None)

  # Save COCO JSON file only if create_coco is True
  if create_coco and coco_writer:
    output_path = os.path.join(INPUT_DIR, COCO_OUTPUT_PATH)
    coco_writer.save(output_path)
    stats = coco_writer.get_statistics()
    print(
        f"\n✅ COCO JSON saved to '{COCO_OUTPUT_PATH}'"
        f" ({stats['num_images']} images,"
        f" {stats['num_annotations']} annotations)."
    )

  print(f"✅ Cropped images saved to '{CLASSIFICATION_DIR}'.")


if __name__ == "__main__":
  app.run(main)
