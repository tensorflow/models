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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf, tf_keras

from official.vision.dataloaders import tf_example_decoder


def _coco91_to_80(classif, box, areas, iscrowds):
  """Function used to reduce COCO 91 to COCO 80 (2017 to 2014 format)."""
  # Vector where index i coralates to the class at index[i].
  class_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
      23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
      44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
      63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
      86, 87, 88, 89, 90
  ]
  new_classes = tf.expand_dims(tf.convert_to_tensor(class_ids), axis=0)

  # Resahpe the classes to in order to build a class mask.
  classes = tf.expand_dims(classif, axis=-1)

  # One hot the classificiations to match the 80 class format.
  ind = classes == tf.cast(new_classes, classes.dtype)

  # Select the max values.
  selected_class = tf.reshape(
      tf.math.argmax(tf.cast(ind, tf.float32), axis=-1), [-1])
  ind = tf.where(tf.reduce_any(ind, axis=-1))

  # Gather the valuable instances.
  classif = tf.gather_nd(selected_class, ind)
  box = tf.gather_nd(box, ind)
  areas = tf.gather_nd(areas, ind)
  iscrowds = tf.gather_nd(iscrowds, ind)

  # Restate the number of viable detections, ideally it should be the same.
  num_detections = tf.shape(classif)[0]
  return classif, box, areas, iscrowds, num_detections


class TfExampleDecoder(tf_example_decoder.TfExampleDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               coco91_to_80=None,
               include_mask=False,
               regenerate_source_id=False,
               mask_binarize_threshold=None):
    """Initialize the example decoder.

    Args:
      coco91_to_80: `bool` indicating whether to convert coco from its 91 class
        format to the 80 class format.
      include_mask: `bool` indicating if the decoder should also decode instance
        masks for instance segmentation.
      regenerate_source_id: `bool` indicating if the source id needs to be
        recreated for each image sample.
      mask_binarize_threshold: `float` for binarizing mask values.
    """
    if coco91_to_80 and include_mask:
      raise ValueError('If masks are included you cannot convert coco from the'
                       '91 class format to the 80 class format.')

    self._coco91_to_80 = coco91_to_80
    super().__init__(
        include_mask=include_mask,
        regenerate_source_id=regenerate_source_id,
        mask_binarize_threshold=mask_binarize_threshold)

  def decode(self, serialized_example):
    """Decode the serialized example.

    Args:
      serialized_example: a single serialized tf.Example string.

    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - source_id: a string scalar tensor.
        - image: a uint8 tensor of shape [None, None, 3].
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    decoded_tensors = super().decode(serialized_example)

    if self._coco91_to_80:
      (decoded_tensors['groundtruth_classes'],
       decoded_tensors['groundtruth_boxes'],
       decoded_tensors['groundtruth_area'],
       decoded_tensors['groundtruth_is_crowd'],
       _) = _coco91_to_80(decoded_tensors['groundtruth_classes'],
                          decoded_tensors['groundtruth_boxes'],
                          decoded_tensors['groundtruth_area'],
                          decoded_tensors['groundtruth_is_crowd'])
    return decoded_tensors
