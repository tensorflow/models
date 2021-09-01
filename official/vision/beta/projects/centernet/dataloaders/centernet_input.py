# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Data parser and processing for Centernet."""

import tensorflow as tf

from official.vision.beta.ops import preprocess_ops
from official.vision.beta.ops import box_ops
from official.vision.beta.dataloaders import parser
from official.vision.beta.dataloaders import utils

from typing import Tuple

CHANNEL_MEANS = (104.01362025, 114.03422265, 119.9165958)
CHANNEL_STDS = (73.6027665, 69.89082075, 70.9150767)


class CenterNetParser(parser.Parser):
  """Parse an image and its annotations into a dictionary of tensors."""
  
  def __init__(self,
               output_width: int = 512,
               output_height: int = 512,
               max_num_instances: int = 128,
               bgr_ordering: bool = True,
               aug_rand_hflip=True,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               aug_rand_saturation=False,
               aug_rand_brightness=False,
               aug_rand_hue=False,
               aug_rand_contrast=False,
               channel_means: Tuple[float, float, float] = CHANNEL_MEANS,
               channel_stds: Tuple[float, float, float] = CHANNEL_STDS,
               dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_width: A `Tensor` or `int` for width of output image.
      output_height: A `Tensor` or `int` for height of output image.
      max_num_instances: An `int` number of maximum number of instances
        in an image.
      bgr_ordering: `bool`, if set will change the channel ordering to be in the
        [blue, red, green] order.
      aug_rand_hflip: `bool`, if True, augment training with random horizontal
        flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      aug_rand_saturation: `bool`, if True, augment training with random
        saturation.
      aug_rand_brightness: `bool`, if True, augment training with random
        brightness.
      aug_rand_hue: `bool`, if True, augment training with random hue.
      aug_rand_hue: `bool`, if True, augment training with random contrast.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    
    Raises:
      Exception: if datatype is not supported.
    """
    self._output_width = output_width
    self._output_height = output_height
    self._max_num_instances = max_num_instances
    self._bgr_ordering = bgr_ordering
    self._channel_means = channel_means
    self._channel_stds = channel_stds
    
    if dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    elif dtype == 'float32':
      self._dtype = tf.float32
    else:
      raise Exception(
          'Unsupported datatype used in parser only '
          '{float16, bfloat16, or float32}')
    
    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_hue = aug_rand_hue
    self._aug_rand_contrast = aug_rand_contrast
  
  def _build_label(self,
                   imshape,
                   boxes,
                   classes,
                   image_info,
                   data):
    
    # Sets up groundtruth data for evaluation.
    groundtruths = {
        'source_id': data['source_id'],
        'height': data['height'],
        'width': data['width'],
        'num_detections': tf.shape(data['groundtruth_classes'])[0],
        'boxes': box_ops.denormalize_boxes(
            data['groundtruth_boxes'], tf.shape(input=data['image'])[0:2]),
        'classes': data['groundtruth_classes'],
        'areas': data['groundtruth_area'],
        'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
    }
    
    groundtruths['source_id'] = utils.process_source_id(
        groundtruths['source_id'])
    groundtruths = utils.pad_groundtruths_to_fixed_size(
        groundtruths, self._max_num_instances)
    
    labels = {
        'boxes': preprocess_ops.clip_or_pad_to_fixed_size(
            boxes, self._max_num_instances, -1),
        'classes': preprocess_ops.clip_or_pad_to_fixed_size(
            classes, self._max_num_instances, -1),
        'image_info': image_info,
        'true_image_shapes': tf.concat([image_info[4, :],
                                        tf.constant([imshape[-1], ],
                                                    dtype=tf.float32)],
                                       axis=0),
        'groundtruths': groundtruths
    }
    
    return labels
  
  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.
    
    We use random flip, random scaling (between 0.6 to 1.3), cropping,
    and color jittering as data augmentation

    Args:
        data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
        images: the image tensor.
        labels: a dict of Tensors that contains labels.
    """
    
    image = tf.cast(data['image'], dtype=tf.float32)
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    
    image_shape = tf.shape(input=image)[0:2]
    
    if self._aug_rand_hflip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)
    
    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)
    
    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        [self._output_height, self._output_width],
        padded_size=[self._output_height, self._output_width],
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    
    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    
    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    
    # Color and lighting jittering
    if self._aug_rand_hue:
      image = tf.image.random_hue(
          image=image, max_delta=.02)
    if self._aug_rand_contrast:
      image = tf.image.random_contrast(
          image=image, lower=0.8, upper=1.25)
    if self._aug_rand_saturation:
      image = tf.image.random_saturation(
          image=image, lower=0.8, upper=1.25)
    if self._aug_rand_brightness:
      image = tf.image.random_brightness(
          image=image, max_delta=.2)
    
    labels = self._build_label(
        imshape=image.get_shape().as_list(),
        boxes=boxes,
        classes=classes,
        image_info=image_info,
        data=data)
    
    if self._bgr_ordering:
      red, green, blue = tf.unstack(image, num=3, axis=2)
      image = tf.stack([blue, green, red], axis=2)

    image = preprocess_ops.normalize_image(
        image=image,
        offset=self._channel_means,
        scale=self._channel_stds)
    
    image = tf.cast(image, self._dtype)
    
    return image, labels
  
  def _parse_eval_data(self, data):
    """Generates images and labels that are usable for model evaluation.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    image = tf.cast(data['image'], dtype=tf.float32)
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    
    image_shape = tf.shape(input=image)[0:2]
    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)
    
    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        [self._output_height, self._output_width],
        padded_size=[self._output_height, self._output_width],
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    
    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    
    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    
    labels = self._build_label(
        imshape=image.get_shape().as_list(),
        boxes=boxes,
        classes=classes,
        image_info=image_info,
        data=data)

    if self._bgr_ordering:
      red, green, blue = tf.unstack(image, num=3, axis=2)
      image = tf.stack([blue, green, red], axis=2)

    image = preprocess_ops.normalize_image(
        image=image,
        offset=self._channel_means,
        scale=self._channel_stds)
    
    image = tf.cast(image, self._dtype)
    
    return image, labels
