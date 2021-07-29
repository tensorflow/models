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


import tensorflow as tf

from official.vision.beta.ops import preprocess_ops
from official.vision.beta.projects.centernet.ops import \
  preprocess_ops as centernet_preprocess_ops
from official.vision.beta.dataloaders import parser, utils

from typing import List

CHANNEL_MEANS = [104.01362025, 114.03422265, 119.9165958]
CHANNEL_STDS = [73.6027665, 69.89082075, 70.9150767]


class CenterNetParser(parser.Parser):
  """ Parser to parse an image and its annotations
  into a dictionary of tensors.
  """
  
  def __init__(self,
               image_w: int = 512,
               image_h: int = 512,
               max_num_instances: int = 128,
               bgr_ordering: bool = True,
               aug_rand_hflip=True,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               aug_rand_saturation=False,
               aug_rand_brightness=False,
               aug_rand_zoom=False,
               aug_rand_hue=False,
               channel_means: List[float] = CHANNEL_MEANS,
               channel_stds: List[float] = CHANNEL_STDS,
               dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      image_w: A `Tensor` or `int` for width of input image.
      image_h: A `Tensor` or `int` for height of input image.
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
      aug_rand_zoom: `bool`, if True, augment training with random zoom.
      aug_rand_hue: `bool`, if True, augment training with random hue.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    self._image_w = image_w
    self._image_h = image_h
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
    self._aug_rand_zoom = aug_rand_zoom
    self._aug_rand_hue = aug_rand_hue
  
  def _build_label(self,
                   image,
                   boxes,
                   image_info,
                   data):
    
    imshape = image.get_shape().as_list()
    height, width = imshape[0:2]
    imshape[-1] = 3
    image.set_shape(imshape)
    
    bshape = boxes.get_shape().as_list()
    boxes = centernet_preprocess_ops.pad_max_instances(
        boxes, self._max_num_instances, 0)
    bshape[0] = self._max_num_instances
    boxes.set_shape(bshape)
    
    classes = data['groundtruth_classes']
    cshape = classes.get_shape().as_list()
    classes = centernet_preprocess_ops.pad_max_instances(
        classes, self._max_num_instances, -1)
    cshape[0] = self._max_num_instances
    classes.set_shape(cshape)
    
    area = data['groundtruth_area']
    ashape = area.get_shape().as_list()
    area = centernet_preprocess_ops.pad_max_instances(
        area, self._max_num_instances, 0)
    ashape[0] = self._max_num_instances
    area.set_shape(ashape)
    
    is_crowd = data['groundtruth_is_crowd']
    ishape = is_crowd.get_shape().as_list()
    is_crowd = centernet_preprocess_ops.pad_max_instances(
        tf.cast(is_crowd, tf.int32), self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    is_crowd.set_shape(ishape)
    
    labels = {
        'source_id': utils.process_source_id(data['source_id']),
        'bbox': boxes,
        'classes': classes,
        'area': area,
        'is_crowd': is_crowd,
        'width': width,
        'height': height,
        'image_info': image_info,
        'num_detections': tf.shape(data['groundtruth_classes'])[0]
    }
    
    return image, labels
  
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

    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    
    if self._aug_rand_hflip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)
    
    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        [self._image_h, self._image_w],
        padded_size=[self._image_h, self._image_w],
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    
    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    
    # Color and lighting jittering
    if self._aug_rand_brightness:
      image = tf.image.random_brightness(
          image=image, max_delta=.1)  # Brightness
    if self._aug_rand_saturation:
      image = tf.image.random_saturation(
          image=image, lower=0.75, upper=1.25)  # Saturation
    if self._aug_rand_hue:
      image = tf.image.random_hue(image=image, max_delta=.3)  # Hue
    
    image, labels = self._build_label(
        image=image,
        boxes=boxes,
        image_info=image_info,
        data=data
    )
    
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
    
    if self._bgr_ordering:
      red, green, blue = tf.unstack(image, num=3, axis=2)
      image = tf.stack([blue, green, red], axis=2)
    
    image = preprocess_ops.normalize_image(
        image=image,
        offset=self._channel_means,
        scale=self._channel_stds)
    
    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        [self._image_h, self._image_w],
        padded_size=[self._image_h, self._image_w],
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image_height, image_width, _ = image.get_shape().as_list()
    
    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    
    image, labels = self._build_label(
        image=image,
        boxes=boxes,
        image_info=image_info,
        data=data
    )
    
    image = tf.cast(image, self._dtype)
    
    return image, labels
