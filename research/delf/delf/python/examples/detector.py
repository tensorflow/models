# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Module to construct object detector function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def MakeDetector(model_dir):
  """Creates a function to detect objects in an image.

  Args:
    model_dir: Directory where SavedModel is located.

  Returns:
    Function that receives an image and returns detection results.
  """
  model = tf.saved_model.load(model_dir)

  # Input and output tensors.
  feeds = ['input_images:0']
  fetches = ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0']

  model = model.prune(feeds=feeds, fetches=fetches)

  def DetectorFn(images):
    """Receives an image and returns detected boxes.

    Args:
      images: Uint8 array with shape (batch, height, width 3) containing a batch
        of RGB images.

    Returns:
      Tuple (boxes, scores, class_indices).
    """
    boxes, scores, class_indices = model(tf.convert_to_tensor(images))

    return boxes.numpy(), scores.numpy(), class_indices.numpy()

  return DetectorFn
