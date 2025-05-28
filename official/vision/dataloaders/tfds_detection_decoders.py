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

"""TFDS detection decoders."""

import tensorflow as tf, tf_keras
from official.vision.dataloaders import decoder


class MSCOCODecoder(decoder.Decoder):
  """A tf.Example decoder for tfds coco datasets."""

  def decode(self, serialized_example):
    """Decode the serialized example.

    Args:
      serialized_example: a dictionary example produced by tfds.

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
    """

    decoded_tensors = {
        'source_id': tf.strings.as_string(serialized_example['image/id']),
        'image': serialized_example['image'],
        'height': tf.cast(tf.shape(serialized_example['image'])[0], tf.int64),
        'width': tf.cast(tf.shape(serialized_example['image'])[1], tf.int64),
        'groundtruth_classes': serialized_example['objects']['label'],
        'groundtruth_is_crowd': serialized_example['objects']['is_crowd'],
        'groundtruth_area': tf.cast(
            serialized_example['objects']['area'], tf.float32),
        'groundtruth_boxes': serialized_example['objects']['bbox'],
    }
    return decoded_tensors


TFDS_ID_TO_DECODER_MAP = {
    'coco/2017': MSCOCODecoder,
    'coco/2014': MSCOCODecoder,
    'coco': MSCOCODecoder,
    'scenic:objects365': MSCOCODecoder,
}
