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
"""Supporting functions for data loading."""

import numpy as np
from PIL import Image

import tensorflow as tf
from delf import utils as image_loading_utils


def pil_imagenet_loader(path, imsize, bounding_box=None, preprocess=True):
  """Pillow loader for the images.

  Args:
    path: Path to image to be loaded.
    imsize: Integer, defines the maximum size of longer image side.
    bounding_box: (x1,y1,x2,y2) tuple to crop the query image.
    preprocess: Bool, whether to preprocess the images in respect to the
      ImageNet dataset.

  Returns:
    image: `Tensor`, image in ImageNet suitable format.
  """
  img = image_loading_utils.RgbLoader(path)

  if bounding_box is not None:
    imfullsize = max(img.size)
    img = img.crop(bounding_box)
    imsize = imsize * max(img.size) / imfullsize

  # Unlike `resize`, `thumbnail` resizes to the largest size that preserves
  # the aspect ratio, making sure that the output image does not exceed the
  # original image size and the size specified in the arguments of thumbnail.
  img.thumbnail((imsize, imsize), Image.ANTIALIAS)
  img = np.array(img)

  if preprocess:
    # Preprocessing for ImageNet data. Converts the images from RGB to BGR,
    # then zero-centers each color channel with respect to the ImageNet
    # dataset, without scaling.
    tf.keras.applications.imagenet_utils.preprocess_input(img, mode='caffe')

  return img


def default_loader(path, imsize, bounding_box=None, preprocess=True):
  """Default loader for the images is using Pillow.

  Args:
    path: Path to image to be loaded.
    imsize: Integer, defines the maximum size of longer image side.
    bounding_box: (x1,y1,x2,y2) tuple to crop the query image.
    preprocess: Bool, whether to preprocess the images in respect to the
      ImageNet dataset.

  Returns:
    image: `Tensor`, image in ImageNet suitable format.
  """
  img = pil_imagenet_loader(path, imsize, bounding_box, preprocess)
  return img
