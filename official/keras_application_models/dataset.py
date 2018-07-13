# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Prepare dataset for keras model benchmark."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Default values for dataset.
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000


def _get_default_image_size(model):
  """Provide default image size for each model."""
  image_size = (224, 224)
  if model in ["inception", "xception", "inceptionresnet"]:
    image_size = (299, 299)
  elif model in ["nasnetlarge"]:
    image_size = (331, 331)
  return image_size


def generate_synthetic_input_dataset(model, num_imgs):
  """Generate synthetic dataset."""
  image_size = _get_default_image_size(model)
  input_shape = (num_imgs,) + image_size + (_NUM_CHANNELS,)

  images = tf.zeros(input_shape, dtype=tf.float32)
  labels = tf.zeros((num_imgs, _NUM_CLASSES), dtype=tf.float32)

  return tf.data.Dataset.from_tensors((images, labels)).repeat()
