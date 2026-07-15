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

"""Image loading, resizing, and general system utilities for the SAM3/DINOv3 tracking pipeline."""

from collections.abc import Sequence
import gc
import pathlib

import cv2
from PIL import Image
import torch


class ImageReadError(OSError):
  """Raised when an image cannot be read from disk.

  Inherits from OSError so callers that broadly catch OSError (e.g. for I/O
  errors) still catch this, while callers who want to be specific can catch
  ImageReadError.
  """


def load_and_resize_image(
    image_path: pathlib.Path, max_short_side: int
) -> Image.Image:
  """Loads an image via OpenCV and downscales it using Lanczos interpolation.

  Lanczos is chosen to match the resampling used by the previous pipeline
  (PIL.Image.LANCZOS), keeping SAM3 inputs close to the previous
  implementation. The two implementations are not bit-identical (PIL and
  OpenCV use slightly different Lanczos filter parameters) but are visually
  equivalent and produce nearly identical downstream detections.

  Args:
    image_path: Path to the image file on disk.
    max_short_side: Maximum allowed length of the shortest side. If the image is
      larger, it is downscaled preserving aspect ratio.

  Returns:
    The loaded and (optionally) resized image as a PIL RGB Image.

  Raises:
    ImageReadError: If OpenCV is unable to decode the file (missing, corrupt,
      or unsupported format).
  """
  bgr_image = cv2.imread(str(image_path))
  if bgr_image is None:
    raise ImageReadError(
        f"OpenCV could not read the image at {image_path!s}. The file may be "
        "missing, corrupt, or in an unsupported format."
    )
  height, width = bgr_image.shape[:2]
  new_width, new_height = _compute_resized_dimensions(
      width=width, height=height, max_short_side=max_short_side
  )
  if (new_width, new_height) != (width, height):
    bgr_resized = cv2.resize(
        bgr_image,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
  else:
    bgr_resized = bgr_image
  rgb_resized = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
  return Image.fromarray(rgb_resized)


def ensure_directories_exist(directories: Sequence[pathlib.Path]) -> None:
  """Creates each directory in the given sequence if it does not already exist.

  Args:
    directories: Paths to create. Each path must be non-empty and not point to
      the current working directory (`.`).

  Raises:
    ValueError: If any path evaluates to `.` (e.g., from `pathlib.Path("")`) or
      is empty (indicates a configuration bug).
  """
  for directory in directories:
    if directory == pathlib.Path("."):
      raise ValueError(
          "ensure_directories_exist received an empty path or the current "
          "working directory ('.'); this usually indicates a configuration "
          "error."
      )
    directory.mkdir(parents=True, exist_ok=True)


def release_caches() -> None:
  """Runs Python garbage collection and empties the CUDA allocator cache.

  Intended to be called between subfolders in a long pipeline run to release
  RAM and VRAM held by unreferenced objects and cached allocations. This does
  not free memory held by live references.
  """
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def _compute_resized_dimensions(
    width: int, height: int, max_short_side: int
) -> tuple[int, int]:
  """Returns the (width, height) that keeps the short side <= `max_short_side`.

  If the image is already small enough, the original dimensions are returned
  unchanged.

  Args:
    width: Original image width in pixels.
    height: Original image height in pixels.
    max_short_side: Maximum allowed length of the shorter side.

  Returns:
    (new_width, new_height), preserving the original aspect ratio.
  """
  short_side = min(width, height)
  if short_side <= max_short_side:
    return width, height
  scale = max_short_side / short_side
  return int(width * scale), int(height * scale)
