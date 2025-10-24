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

"""Utilities for batched I/O operations to improve pipeline performance."""

from concurrent import futures
import os
from typing import List
import cv2
import numpy as np

ThreadPoolExecutor = futures.ThreadPoolExecutor


def save_masked_objects(
    image: np.ndarray,
    masks: List[np.ndarray],
    boxes: List[np.ndarray],
    source_image_path: str,
    output_dir: str,
) -> List[str]:
  """Saves a batch of masked objects.

  This function is called by BatchedMaskWriter's thread pool, so parallelism
  is handled at the image level by the outer executor. Each call processes
  one image's worth of objects sequentially.

  Args:
    image: Source image.
    masks: Binary masks to save.
    boxes: Bounding boxes corresponding to masks.
    source_image_path: Path to the original image file.
    output_dir: Directory path to save cropped objects.

  Returns:
    Paths to saved images.
  """
  image_name = os.path.splitext(os.path.basename(source_image_path))[0]

  files_to_save = []
  for idx, (mask, box) in enumerate(zip(masks, boxes)):
    try:
      masked_object = extract_masked_object(image, mask, box)
      output_path = os.path.join(output_dir, f"{image_name}_object_{idx}.png")
      result = cv2.imwrite(output_path, masked_object)
      if result:
        files_to_save.append(output_path)
    except (ValueError, SystemError, AttributeError) as e:
      print(f"[ERROR] Skipped object {idx}: {e}")
      continue

  return files_to_save


def extract_masked_object(
    image: np.ndarray, mask: np.ndarray, box: np.ndarray
) -> np.ndarray:
  """Extracts object from image using mask and bounding box.

  Args:
    image: Source image (H, W, 3).
    mask: Binary mask (H, W).
    box: Bounding box [x1, y1, x2, y2].

  Returns:
    Cropped and masked object as numpy array.
  """

  x1, y1, x2, y2 = map(int, box)

  # Crop the image and mask
  cropped_image = image[y1:y2, x1:x2]
  cropped_mask = mask[y1:y2, x1:x2]
  cropped_mask = cropped_mask.astype(bool)
  masked_object = cropped_image.copy()
  masked_object[~cropped_mask] = 1  # Black for non-masked pixels (background)
  masked_object_rgb = cv2.cvtColor(masked_object, cv2.COLOR_BGR2RGB)

  return masked_object_rgb


class BatchedMaskWriter:
  """A thread pool manager to parallelize object saving from images.

  Each worker takes an image and then sequentially writes all of its detected
  mask objects.
  """

  def __init__(self, output_dir: str):
    self.output_dir = output_dir
    self.executor = ThreadPoolExecutor(max_workers=4)
    self.futures = []
    os.makedirs(output_dir, exist_ok=True)

  def __exit__(self, exc_type, exc_val, exc_tb):
    # Wait for all writes to complete
    for future in self.futures:
      future.result()
    self.executor.shutdown(wait=True)

  def add_batch(
      self,
      image: np.ndarray,
      masks: List[np.ndarray],
      boxes: List[np.ndarray],
      source_path: str,
  ):
    """Add a batch of masks to be written asynchronously."""
    future = self.executor.submit(
        save_masked_objects,
        image,
        masks,
        boxes,
        source_path,
        self.output_dir,
    )
    self.futures.append(future)
