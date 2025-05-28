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

from official.vision.dataloaders import decoder


def _generate_source_id(image_bytes):
  # Hashing using 22 bits since float32 has only 23 mantissa bits.
  return tf.strings.as_string(
      tf.strings.to_hash_bucket_fast(image_bytes, 2 ** 22 - 1))


class TfExampleDecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""

  def __init__(
      self,
      include_mask=False,
      regenerate_source_id=False,
      mask_binarize_threshold=None,
      attribute_names=None,
  ):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id
    self._keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/height': tf.io.FixedLenFeature((), tf.int64, -1),
        'image/width': tf.io.FixedLenFeature((), tf.int64, -1),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/area': tf.io.VarLenFeature(tf.float32),
        'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
    }
    attribute_names = attribute_names or []
    for attr_name in attribute_names:
      self._keys_to_features[f'image/object/attribute/{attr_name}'] = (
          tf.io.VarLenFeature(tf.int64)
      )
    self._attribute_names = attribute_names

    self._mask_binarize_threshold = mask_binarize_threshold
    if include_mask:
      self._keys_to_features.update({
          'image/object/mask': tf.io.VarLenFeature(tf.string),
      })
    if not regenerate_source_id:
      self._keys_to_features.update({
          'image/source_id': tf.io.FixedLenFeature((), tf.string),
      })

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    return image

  def _decode_boxes(self, parsed_tensors):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

  def _decode_classes(self, parsed_tensors):
    return parsed_tensors['image/object/class/label']

  def _decode_attributes(self, parsed_tensors):
    attribute_dict = dict()
    for attr_name in self._attribute_names:
      attr_array = parsed_tensors[f'image/object/attribute/{attr_name}']
      # TODO(b/269654135): Support decoding of fully 2D attributes.
      attribute_dict[attr_name] = tf.expand_dims(attr_array, -1)
    return attribute_dict

  def _decode_areas(self, parsed_tensors):
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    height = tf.cast(parsed_tensors['image/height'], dtype=tf.float32)
    width = tf.cast(parsed_tensors['image/width'], dtype=tf.float32)
    return tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/area'])[0], 0),
        lambda: parsed_tensors['image/object/area'],
        lambda: (xmax - xmin) * (ymax - ymin) * height * width)

  def _decode_masks(self, parsed_tensors):
    """Decode a set of PNG masks to the tf.float32 tensors."""

    def _decode_png_mask(png_bytes):
      mask = tf.squeeze(
          tf.io.decode_png(png_bytes, channels=1, dtype=tf.uint8), axis=-1)
      mask = tf.cast(mask, dtype=tf.float32)
      mask.set_shape([None, None])
      return mask

    height = parsed_tensors['image/height']
    width = parsed_tensors['image/width']
    masks = parsed_tensors['image/object/mask']
    return tf.cond(
        pred=tf.greater(tf.size(input=masks), 0),
        true_fn=lambda: tf.map_fn(_decode_png_mask, masks, dtype=tf.float32),
        false_fn=lambda: tf.zeros([0, height, width], dtype=tf.float32))

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
    parsed_tensors = tf.io.parse_single_example(
        serialized=serialized_example, features=self._keys_to_features)
    for k in parsed_tensors:
      if isinstance(parsed_tensors[k], tf.SparseTensor):
        if parsed_tensors[k].dtype == tf.string:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value='')
        else:
          parsed_tensors[k] = tf.sparse.to_dense(
              parsed_tensors[k], default_value=0)

    if self._regenerate_source_id:
      source_id = _generate_source_id(parsed_tensors['image/encoded'])
    else:
      source_id = tf.cond(
          tf.greater(tf.strings.length(parsed_tensors['image/source_id']), 0),
          lambda: parsed_tensors['image/source_id'],
          lambda: _generate_source_id(parsed_tensors['image/encoded']))
    image = self._decode_image(parsed_tensors)
    boxes = self._decode_boxes(parsed_tensors)
    classes = self._decode_classes(parsed_tensors)
    areas = self._decode_areas(parsed_tensors)

    attributes = self._decode_attributes(parsed_tensors)

    decode_image_shape = tf.logical_or(
        tf.equal(parsed_tensors['image/height'], -1),
        tf.equal(parsed_tensors['image/width'], -1))
    image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

    parsed_tensors['image/height'] = tf.where(decode_image_shape,
                                              image_shape[0],
                                              parsed_tensors['image/height'])
    parsed_tensors['image/width'] = tf.where(decode_image_shape, image_shape[1],
                                             parsed_tensors['image/width'])

    is_crowds = tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/is_crowd'])[0], 0),
        lambda: tf.cast(parsed_tensors['image/object/is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(classes, dtype=tf.bool))
    if self._include_mask:
      masks = self._decode_masks(parsed_tensors)

      if self._mask_binarize_threshold is not None:
        masks = tf.cast(masks > self._mask_binarize_threshold, tf.float32)

    decoded_tensors = {
        'source_id': source_id,
        'image': image,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'groundtruth_classes': classes,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    if self._attribute_names:
      decoded_tensors.update({'groundtruth_attributes': attributes})
    if self._include_mask:
      decoded_tensors.update({
          'groundtruth_instance_masks': masks,
          'groundtruth_instance_masks_png': parsed_tensors['image/object/mask'],
      })
    return decoded_tensors
