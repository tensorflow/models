# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""COCO data loader for DETR."""

from typing import Tuple
import tensorflow as tf, tf_keras

from official.vision.dataloaders import parser

from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops

RESIZE_SCALES = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)


class Parser(parser.Parser):
  """Parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               class_offset: int = 0,
               output_size: Tuple[int, int] = (1333, 1333),
               max_num_boxes: int = 100,
               resize_scales: Tuple[int, ...] = RESIZE_SCALES,
               aug_rand_hflip=True):
    self._class_offset = class_offset
    self._output_size = output_size
    self._max_num_boxes = max_num_boxes
    self._resize_scales = resize_scales
    self._aug_rand_hflip = aug_rand_hflip

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    classes = data['groundtruth_classes'] + self._class_offset
    boxes = data['groundtruth_boxes']
    is_crowd = data['groundtruth_is_crowd']

    # Gets original image.
    image = data['image']

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)
    image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    do_crop = tf.greater(tf.random.uniform([]), 0.5)
    if do_crop:
      # Rescale
      boxes = box_ops.denormalize_boxes(boxes, tf.shape(image)[:2])
      index = tf.random.categorical(tf.zeros([1, 3]), 1)[0]
      scales = tf.gather([400.0, 500.0, 600.0], index, axis=0)
      short_side = scales[0]
      image, image_info = preprocess_ops.resize_image(image, short_side)
      boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_info[2, :],
                                                   image_info[1, :],
                                                   image_info[3, :])
      boxes = box_ops.normalize_boxes(boxes, image_info[1, :])

      # Do croping
      shape = tf.cast(image_info[1], dtype=tf.int32)
      h = tf.random.uniform([],
                            384,
                            tf.math.minimum(shape[0], 600),
                            dtype=tf.int32)
      w = tf.random.uniform([],
                            384,
                            tf.math.minimum(shape[1], 600),
                            dtype=tf.int32)
      i = tf.random.uniform([], 0, shape[0] - h + 1, dtype=tf.int32)
      j = tf.random.uniform([], 0, shape[1] - w + 1, dtype=tf.int32)
      image = tf.image.crop_to_bounding_box(image, i, j, h, w)
      boxes = tf.clip_by_value(
          (boxes[..., :] * tf.cast(
              tf.stack([shape[0], shape[1], shape[0], shape[1]]),
              dtype=tf.float32) -
           tf.cast(tf.stack([i, j, i, j]), dtype=tf.float32)) /
          tf.cast(tf.stack([h, w, h, w]), dtype=tf.float32), 0.0, 1.0)
    scales = tf.constant(self._resize_scales, dtype=tf.float32)
    index = tf.random.categorical(tf.zeros([1, 11]), 1)[0]
    scales = tf.gather(scales, index, axis=0)

    image_shape = tf.shape(image)[:2]
    boxes = box_ops.denormalize_boxes(boxes, image_shape)
    short_side = scales[0]
    image, image_info = preprocess_ops.resize_image(image, short_side,
                                                    max(self._output_size))
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_info[2, :],
                                                 image_info[1, :],
                                                 image_info[3, :])
    boxes = box_ops.normalize_boxes(boxes, image_info[1, :])

    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    is_crowd = tf.gather(is_crowd, indices)
    boxes = box_ops.yxyx_to_cycxhw(boxes)

    image = tf.image.pad_to_bounding_box(image, 0, 0, self._output_size[0],
                                         self._output_size[1])
    labels = {
        'classes':
            preprocess_ops.clip_or_pad_to_fixed_size(classes,
                                                     self._max_num_boxes),
        'boxes':
            preprocess_ops.clip_or_pad_to_fixed_size(boxes, self._max_num_boxes)
    }

    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    is_crowd = data['groundtruth_is_crowd']

    # Gets original image and its size.
    image = data['image']

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    scales = tf.constant([self._resize_scales[-1]], tf.float32)

    image_shape = tf.shape(image)[:2]
    boxes = box_ops.denormalize_boxes(boxes, image_shape)
    gt_boxes = boxes
    short_side = scales[0]
    image, image_info = preprocess_ops.resize_image(image, short_side,
                                                    max(self._output_size))
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_info[2, :],
                                                 image_info[1, :],
                                                 image_info[3, :])
    boxes = box_ops.normalize_boxes(boxes, image_info[1, :])

    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    is_crowd = tf.gather(is_crowd, indices)
    boxes = box_ops.yxyx_to_cycxhw(boxes)

    image = tf.image.pad_to_bounding_box(image, 0, 0, self._output_size[0],
                                         self._output_size[1])
    labels = {
        'classes':
            preprocess_ops.clip_or_pad_to_fixed_size(classes,
                                                     self._max_num_boxes),
        'boxes':
            preprocess_ops.clip_or_pad_to_fixed_size(boxes, self._max_num_boxes)
    }
    labels.update({
        'id':
            int(data['source_id']),
        'image_info':
            image_info,
        'is_crowd':
            preprocess_ops.clip_or_pad_to_fixed_size(is_crowd,
                                                     self._max_num_boxes),
        'gt_boxes':
            preprocess_ops.clip_or_pad_to_fixed_size(gt_boxes,
                                                     self._max_num_boxes),
    })

    return image, labels
