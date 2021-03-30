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

"""Detection Data parser and processing for YOLO.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""

import tensorflow as tf

from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.projects.yolo.ops import box_ops as yolo_box_ops
from official.vision.beta.projects.yolo.ops import preprocess_ops as yolo_preprocess_ops


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               num_classes,
               fixed_size=True,
               jitter_im=0.1,
               jitter_boxes=0.005,
               use_tie_breaker=True,
               min_level=3,
               max_level=5,
               masks=None,
               max_process_size=608,
               min_process_size=320,
               max_num_instances=200,
               random_flip=True,
               aug_rand_saturation=True,
               aug_rand_brightness=True,
               aug_rand_zoom=True,
               aug_rand_hue=True,
               anchors=None,
               seed=10,
               dtype=tf.float32):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: a `Tuple` for (width, height) of input image.
      num_classes: a `Tensor` or `int` for the number of classes.
      fixed_size: a `bool` if True all output images have the same size.
      jitter_im: a `float` representing a pixel value that is the maximum jitter
        applied to the image for data augmentation during training.
      jitter_boxes: a `float` representing a pixel value that is the maximum
        jitter applied to the bounding box for data augmentation during
        training.
      use_tie_breaker: boolean value for wether or not to use the tie_breaker.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      masks: a `Tensor`, `List` or `numpy.ndarray` for anchor masks.
      max_process_size: an `int` for maximum image width and height.
      min_process_size: an `int` for minimum image width and height ,
      max_num_instances: an `int` number of maximum number of instances in an
        image.
      random_flip: a `bool` if True, augment training with random horizontal
        flip.
      aug_rand_saturation: `bool`, if True, augment training with random
        saturation.
      aug_rand_brightness: `bool`, if True, augment training with random
        brightness.
      aug_rand_zoom: `bool`, if True, augment training with random zoom.
      aug_rand_hue: `bool`, if True, augment training with random hue.
      anchors: a `Tensor`, `List` or `numpy.ndarrray` for bounding box priors.
      seed: an `int` for the seed used by tf.random
      dtype: a `tf.dtypes.DType` object that represents the dtype the outputs
        will be casted to. The available types are tf.float32, tf.float16, or
        tf.bfloat16.
    """
    self._net_down_scale = 2**max_level

    self._num_classes = num_classes
    self._image_w = (output_size[0] //
                     self._net_down_scale) * self._net_down_scale
    self._image_h = (output_size[1] //
                     self._net_down_scale) * self._net_down_scale

    self._max_process_size = max_process_size
    self._min_process_size = min_process_size
    self._fixed_size = fixed_size

    self._anchors = anchors
    self._masks = {
        key: tf.convert_to_tensor(value) for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker

    self._jitter_im = 0.0 if jitter_im is None else jitter_im
    self._jitter_boxes = 0.0 if jitter_boxes is None else jitter_boxes
    self._max_num_instances = max_num_instances
    self._random_flip = random_flip

    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_zoom = aug_rand_zoom
    self._aug_rand_hue = aug_rand_hue

    self._seed = seed
    self._dtype = dtype

  def _build_grid(self, raw_true, width, batch=False, use_tie_breaker=False):
    mask = self._masks
    for key in self._masks.keys():
      if not batch:
        mask[key] = yolo_preprocess_ops.build_grided_gt(
            raw_true, self._masks[key], width // 2**int(key),
            raw_true['bbox'].dtype, use_tie_breaker)
      else:
        mask[key] = yolo_preprocess_ops.build_batch_grided_gt(
            raw_true, self._masks[key], width // 2**int(key),
            raw_true['bbox'].dtype, use_tie_breaker)
    return mask

  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.

    Args:
      data: a dict of Tensors produced by the decoder.
    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """

    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    width = shape[0]
    height = shape[1]

    image, boxes = yolo_preprocess_ops.fit_preserve_aspect_ratio(
        image,
        boxes,
        width=width,
        height=height,
        target_dim=self._max_process_size)

    image_shape = tf.shape(image)[:2]

    if self._random_flip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    randscale = self._image_w // self._net_down_scale

    if not self._fixed_size:
      do_scale = tf.greater(
          tf.random.uniform([], minval=0, maxval=1, seed=self._seed), 0.5)
      if do_scale:
        # This scales the image to a random multiple of net_down_scale
        # between 320 to 608
        randscale = tf.random.uniform(
            [],
            minval=self._min_process_size // self._net_down_scale,
            maxval=self._max_process_size // self._net_down_scale,
            seed=self._seed,
            dtype=tf.int32) * self._net_down_scale

    if self._jitter_boxes != 0.0:
      boxes = box_ops.denormalize_boxes(boxes, image_shape)
      boxes = box_ops.jitter_boxes(boxes, 0.025)
      boxes = box_ops.normalize_boxes(boxes, image_shape)

    # YOLO loss function uses x-center, y-center format
    boxes = yolo_box_ops.yxyx_to_xcycwh(boxes)

    if self._jitter_im != 0.0:
      image, boxes = yolo_preprocess_ops.random_translate(
          image, boxes, self._jitter_im, seed=self._seed)

    if self._aug_rand_zoom:
      image, boxes = yolo_preprocess_ops.resize_crop_filter(
          image,
          boxes,
          default_width=self._image_w,
          default_height=self._image_h,
          target_width=randscale,
          target_height=randscale)
    image = tf.image.resize(image, (416, 416), preserve_aspect_ratio=False)

    if self._aug_rand_brightness:
      image = tf.image.random_brightness(
          image=image, max_delta=.1)  # Brightness
    if self._aug_rand_saturation:
      image = tf.image.random_saturation(
          image=image, lower=0.75, upper=1.25)  # Saturation
    if self._aug_rand_hue:
      image = tf.image.random_hue(image=image, max_delta=.3)  # Hue
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Find the best anchor for the ground truth labels to maximize the iou
    best_anchors = yolo_preprocess_ops.get_best_anchor(
        boxes, self._anchors, width=self._image_w, height=self._image_h)

    # Padding
    boxes = preprocess_ops.clip_or_pad_to_fixed_size(boxes,
                                                     self._max_num_instances, 0)
    classes = preprocess_ops.clip_or_pad_to_fixed_size(
        data['groundtruth_classes'], self._max_num_instances, -1)
    best_anchors = preprocess_ops.clip_or_pad_to_fixed_size(
        best_anchors, self._max_num_instances, 0)
    area = preprocess_ops.clip_or_pad_to_fixed_size(data['groundtruth_area'],
                                                    self._max_num_instances, 0)
    is_crowd = preprocess_ops.clip_or_pad_to_fixed_size(
        tf.cast(data['groundtruth_is_crowd'], tf.int32),
        self._max_num_instances, 0)

    labels = {
        'source_id': data['source_id'],
        'bbox': tf.cast(boxes, self._dtype),
        'classes': tf.cast(classes, self._dtype),
        'area': tf.cast(area, self._dtype),
        'is_crowd': is_crowd,
        'best_anchors': tf.cast(best_anchors, self._dtype),
        'width': width,
        'height': height,
        'num_detections': tf.shape(data['groundtruth_classes'])[0],
    }

    if self._fixed_size:
      grid = self._build_grid(
          labels, self._image_w, use_tie_breaker=self._use_tie_breaker)
      labels.update({'grid_form': grid})

    return image, labels

  def _parse_eval_data(self, data):
    """Generates images and labels that are usable for model training.

    Args:
      data: a dict of Tensors produced by the decoder.
    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """

    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    width = shape[0]
    height = shape[1]

    image, boxes = yolo_preprocess_ops.fit_preserve_aspect_ratio(
        image, boxes, width=width, height=height, target_dim=self._image_w)
    boxes = yolo_box_ops.yxyx_to_xcycwh(boxes)

    # Find the best anchor for the ground truth labels to maximize the iou
    best_anchors = yolo_preprocess_ops.get_best_anchor(
        boxes, self._anchors, width=self._image_w, height=self._image_h)
    boxes = yolo_preprocess_ops.pad_max_instances(boxes,
                                                  self._max_num_instances, 0)
    classes = yolo_preprocess_ops.pad_max_instances(data['groundtruth_classes'],
                                                    self._max_num_instances, 0)
    best_anchors = yolo_preprocess_ops.pad_max_instances(
        best_anchors, self._max_num_instances, 0)
    area = yolo_preprocess_ops.pad_max_instances(data['groundtruth_area'],
                                                 self._max_num_instances, 0)
    is_crowd = yolo_preprocess_ops.pad_max_instances(
        tf.cast(data['groundtruth_is_crowd'], tf.int32),
        self._max_num_instances, 0)

    labels = {
        'source_id': data['source_id'],
        'bbox': tf.cast(boxes, self._dtype),
        'classes': tf.cast(classes, self._dtype),
        'area': tf.cast(area, self._dtype),
        'is_crowd': is_crowd,
        'best_anchors': tf.cast(best_anchors, self._dtype),
        'width': width,
        'height': height,
        'num_detections': tf.shape(data['groundtruth_classes'])[0],
    }

    grid = self._build_grid(
        labels,
        self._image_w,
        batch=False,
        use_tie_breaker=self._use_tie_breaker)
    labels.update({'grid_form': grid})
    return image, labels

  def _postprocess_fn(self, image, label):
    randscale = self._image_w // self._net_down_scale
    if not self._fixed_size:
      do_scale = tf.greater(
          tf.random.uniform([], minval=0, maxval=1, seed=self._seed), 0.5)
      if do_scale:
        # This scales the image to a random multiple of net_down_scale
        # between 320 to 608
        randscale = tf.random.uniform(
            [],
            minval=self._min_process_size // self._net_down_scale,
            maxval=self._max_process_size // self._net_down_scale,
            seed=self._seed,
            dtype=tf.int32) * self._net_down_scale
    width = randscale
    image = tf.image.resize(image, (width, width))
    grid = self._build_grid(
        label, width, batch=True, use_tie_breaker=self._use_tie_breaker)
    label.update({'grid_form': grid})
    return image, label

  def postprocess_fn(self, is_training=True):
    return self._postprocess_fn if not self._fixed_size and is_training else None
