# Lint as: python3
# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Functions for generic image dataset creation."""

import os

from delf.python.datasets import utils


class ImagesFromList():
  """A generic data loader that loads images from a list.

  Supports images of different sizes.
  """

  def __init__(self, root, image_paths, imsize=None, bounding_boxes=None,
               loader=utils.default_loader):
    """ImagesFromList object initialization.

    Args:
      root: String, root directory path.
      image_paths: List, relative image paths as strings.
      imsize: Integer, defines the maximum size of longer image side.
      bounding_boxes: List of (x1,y1,x2,y2) tuples to crop the query images.
      loader: Callable, a function to load an image given its path.

    Raises:
      ValueError: Raised if `image_paths` list is empty.
    """
    # List of the full image filenames.
    images_filenames = [os.path.join(root, image_path) for image_path in
                        image_paths]

    if not images_filenames:
      raise ValueError("Dataset contains 0 images.")

    self.root = root
    self.images = image_paths
    self.imsize = imsize
    self.images_filenames = images_filenames
    self.bounding_boxes = bounding_boxes
    self.loader = loader

  def __getitem__(self, index):
    """Called to load an image at the given `index`.

    Args:
        index: Integer, image index.

    Returns:
        image: Tensor, loaded image.
    """
    path = self.images_filenames[index]

    if self.bounding_boxes is not None:
      img = self.loader(path, self.imsize, self.bounding_boxes[index])
    else:
      img = self.loader(path, self.imsize)

    return img

  def __len__(self):
    """Implements the built-in function len().

    Returns:
      len: Number of images in the dataset.
    """
    return len(self.images_filenames)
