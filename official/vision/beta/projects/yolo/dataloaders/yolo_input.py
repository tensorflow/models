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

"""Detection Data parser and processing for YOLO."""
import tensorflow as tf

from official.vision.beta.dataloaders import parser
from official.vision.beta.dataloaders import utils
from official.vision.beta.ops import box_ops as bbox_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.projects.yolo.ops import anchor
from official.vision.beta.projects.yolo.ops import preprocessing_ops


class Parser(parser.Parser):
  """Parse the dataset in to the YOLO model format."""

  def __init__(self,
               output_size,
               anchors,
               expanded_strides,
               level_limits=None,
               max_num_instances=200,
               area_thresh=0.1,
               aug_rand_hue=1.0,
               aug_rand_saturation=1.0,
               aug_rand_brightness=1.0,
               letter_box=False,
               random_pad=True,
               random_flip=True,
               jitter=0.0,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               aug_rand_translate=0.0,
               aug_rand_perspective=0.0,
               aug_rand_angle=0.0,
               anchor_t=4.0,
               scale_xy=None,
               best_match_only=False,
               darknet=False,
               use_tie_breaker=True,
               dtype='float32',
               seed=None):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `List` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      anchors: `Dict[List[Union[int, float]]]` of anchor boxes to be bes used in
        each level.
      expanded_strides: `Dict[int]` for how much the model scales down the
        images at the largest level. For example, level 3 down samples the image
        by a factor of 16, in the expanded strides dictionary, we will pass
        along {3: 16} indicating that relative to the original image, the shapes
          must be reduced by a factor of 16 to compute the loss.
      level_limits: `List` the box sizes that will be allowed at each FPN level
        as is done in the FCOS and YOLOX paper for anchor free box assignment.
      max_num_instances: `int` for the number of boxes to compute loss on.
      area_thresh: `float` for the minimum area of a box to allow to pass
        through for optimization.
      aug_rand_hue: `float` indicating the maximum scaling value for hue.
        saturation will be scaled between 1 - value and 1 + value.
      aug_rand_saturation: `float` indicating the maximum scaling value for
        saturation. saturation will be scaled between 1/value and value.
      aug_rand_brightness: `float` indicating the maximum scaling value for
        brightness. brightness will be scaled between 1/value and value.
      letter_box: `boolean` indicating whether upon start of the data pipeline
        regardless of the preprocessing ops that are used, the aspect ratio of
        the images should be preserved.
      random_pad: `bool` indiccating wether to use padding to apply random
        translation, true for darknet yolo false for scaled yolo.
      random_flip: `boolean` indicating whether or not to randomly flip the
        image horizontally.
      jitter: `float` for the maximum change in aspect ratio expected in each
        preprocessing step.
      aug_scale_min: `float` indicating the minimum scaling value for image
        scale jitter.
      aug_scale_max: `float` indicating the maximum scaling value for image
        scale jitter.
      aug_rand_translate: `float` ranging from 0 to 1 indicating the maximum
        amount to randomly translate an image.
      aug_rand_perspective: `float` ranging from 0.000 to 0.001 indicating how
        much to prespective warp the image.
      aug_rand_angle: `float` indicating the maximum angle value for angle.
        angle will be changes between 0 and value.
      anchor_t: `float` indicating the threshold over which an anchor will be
        considered for prediction, at zero, all the anchors will be used and at
        1.0 only the best will be used. for anchor thresholds larger than 1.0 we
        stop using the IOU for anchor comparison and resort directly to
        comparing the width and height, this is used for the scaled models.
      scale_xy: dictionary `float` values inidcating how far each pixel can see
        outside of its containment of 1.0. a value of 1.2 indicates there is a
        20% extended radius around each pixel that this specific pixel can
        predict values for a center at. the center can range from 0 - value/2 to
        1 + value/2, this value is set in the yolo filter, and resused here.
        there should be one value for scale_xy for each level from min_level to
        max_level.
      best_match_only: `boolean` indicating how boxes are selected for
        optimization.
      darknet: `boolean` indicating which data pipeline to use. Setting to True
        swaps the pipeline to output images realtive to Yolov4 and older.
      use_tie_breaker: `boolean` indicating whether to use the anchor threshold
        value.
      dtype: `str` indicating the output datatype of the datapipeline selecting
        from {"float32", "float16", "bfloat16"}.
      seed: `int` the seed for random number generation.
    """
    for key in anchors:
      # Assert that the width and height is viable
      assert output_size[1] % expanded_strides[str(key)] == 0
      assert output_size[0] % expanded_strides[str(key)] == 0

    # Set the width and height properly and base init:
    self._image_w = output_size[1]
    self._image_h = output_size[0]
    self._max_num_instances = max_num_instances

    # Image scaling params
    self._jitter = 0.0 if jitter is None else jitter
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._aug_rand_translate = aug_rand_translate
    self._aug_rand_perspective = aug_rand_perspective

    # Image spatial distortion
    self._random_flip = random_flip
    self._letter_box = letter_box
    self._random_pad = random_pad
    self._aug_rand_angle = aug_rand_angle

    # Color space distortion of the image
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_hue = aug_rand_hue

    # Set the per level values needed for operation
    self._darknet = darknet
    self._area_thresh = area_thresh
    self._level_limits = level_limits

    self._seed = seed
    self._dtype = dtype

    self._label_builder = anchor.YoloAnchorLabeler(
        anchors=anchors,
        anchor_free_level_limits=level_limits,
        level_strides=expanded_strides,
        center_radius=scale_xy,
        max_num_instances=max_num_instances,
        match_threshold=anchor_t,
        best_matches_only=best_match_only,
        use_tie_breaker=use_tie_breaker,
        darknet=darknet,
        dtype=dtype)

  def _pad_infos_object(self, image):
    """Get a Tensor to pad the info object list."""
    shape_ = tf.shape(image)
    val = tf.stack([
        tf.cast(shape_[:2], tf.float32),
        tf.cast(shape_[:2], tf.float32),
        tf.ones_like(tf.cast(shape_[:2], tf.float32)),
        tf.zeros_like(tf.cast(shape_[:2], tf.float32)),
    ])
    return val

  def _jitter_scale(self, image, shape, letter_box, jitter, random_pad,
                    aug_scale_min, aug_scale_max, translate, angle,
                    perspective):
    """Distort and scale each input image."""
    infos = []
    if (aug_scale_min != 1.0 or aug_scale_max != 1.0):
      crop_only = True
      # jitter gives you only one info object, resize and crop gives you one,
      # if crop only then there can be 1 form jitter and 1 from crop
      infos.append(self._pad_infos_object(image))
    else:
      crop_only = False
    image, crop_info, _ = preprocessing_ops.resize_and_jitter_image(
        image,
        shape,
        letter_box=letter_box,
        jitter=jitter,
        crop_only=crop_only,
        random_pad=random_pad,
        seed=self._seed,
    )
    infos.extend(crop_info)
    image, _, affine = preprocessing_ops.affine_warp_image(
        image,
        shape,
        scale_min=aug_scale_min,
        scale_max=aug_scale_max,
        translate=translate,
        degrees=angle,
        perspective=perspective,
        random_pad=random_pad,
        seed=self._seed,
    )
    return image, infos, affine

  def _parse_train_data(self, data):
    """Parses data for training."""

    # Initialize the shape constants.
    image = data['image']
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    if self._random_flip:
      # Randomly flip the image horizontally.
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    if not data['is_mosaic']:
      image, infos, affine = self._jitter_scale(
          image, [self._image_h, self._image_w], self._letter_box, self._jitter,
          self._random_pad, self._aug_scale_min, self._aug_scale_max,
          self._aug_rand_translate, self._aug_rand_angle,
          self._aug_rand_perspective)

      # Clip and clean boxes.
      boxes, inds = preprocessing_ops.transform_and_clip_boxes(
          boxes,
          infos,
          affine=affine,
          shuffle_boxes=False,
          area_thresh=self._area_thresh,
          augment=True,
          seed=self._seed)
      classes = tf.gather(classes, inds)
      info = infos[-1]
    else:
      image = tf.image.resize(
          image, (self._image_h, self._image_w), method='nearest')
      output_size = tf.cast([640, 640], tf.float32)
      boxes_ = bbox_ops.denormalize_boxes(boxes, output_size)
      inds = bbox_ops.get_non_empty_box_indices(boxes_)
      boxes = tf.gather(boxes, inds)
      classes = tf.gather(classes, inds)
      info = self._pad_infos_object(image)

    # Apply scaling to the hue saturation and brightness of an image.
    image = tf.cast(image, dtype=self._dtype)
    image = image / 255.0
    image = preprocessing_ops.image_rand_hsv(
        image,
        self._aug_rand_hue,
        self._aug_rand_saturation,
        self._aug_rand_brightness,
        seed=self._seed,
        darknet=self._darknet or self._level_limits is not None)

    # Cast the image to the selcted datatype.
    image, labels = self._build_label(
        image, boxes, classes, info, inds, data, is_training=True)
    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for evaluation."""

    # Get the image shape constants and cast the image to the selcted datatype.
    image = tf.cast(data['image'], dtype=self._dtype)
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, infos, _ = preprocessing_ops.resize_and_jitter_image(
        image, [self._image_h, self._image_w],
        letter_box=self._letter_box,
        random_pad=False,
        shiftx=0.5,
        shifty=0.5,
        jitter=0.0)

    # Clip and clean boxes.
    image = image / 255.0
    boxes, inds = preprocessing_ops.transform_and_clip_boxes(
        boxes, infos, shuffle_boxes=False, area_thresh=0.0, augment=True)
    classes = tf.gather(classes, inds)
    info = infos[-1]

    image, labels = self._build_label(
        image, boxes, classes, info, inds, data, is_training=False)
    return image, labels

  def set_shape(self, values, pad_axis=0, pad_value=0, inds=None):
    """Calls set shape for all input objects."""
    if inds is not None:
      values = tf.gather(values, inds)
    vshape = values.get_shape().as_list()

    values = preprocessing_ops.pad_max_instances(
        values, self._max_num_instances, pad_axis=pad_axis, pad_value=pad_value)

    vshape[pad_axis] = self._max_num_instances
    values.set_shape(vshape)
    return values

  def _build_label(self,
                   image,
                   gt_boxes,
                   gt_classes,
                   info,
                   inds,
                   data,
                   is_training=True):
    """Label construction for both the train and eval data."""
    width = self._image_w
    height = self._image_h

    # Set the image shape.
    imshape = image.get_shape().as_list()
    imshape[-1] = 3
    image.set_shape(imshape)

    labels = dict()
    (labels['inds'], labels['upds'],
     labels['true_conf']) = self._label_builder(gt_boxes, gt_classes, width,
                                                height)

    # Set/fix the boxes shape.
    boxes = self.set_shape(gt_boxes, pad_axis=0, pad_value=0)
    classes = self.set_shape(gt_classes, pad_axis=0, pad_value=-1)

    # Build the dictionary set.
    labels.update({
        'source_id': utils.process_source_id(data['source_id']),
        'bbox': tf.cast(boxes, dtype=self._dtype),
        'classes': tf.cast(classes, dtype=self._dtype),
    })

    # Update the labels dictionary.
    if not is_training:

      # Sets up groundtruth data for evaluation.
      groundtruths = {
          'source_id': labels['source_id'],
          'height': height,
          'width': width,
          'num_detections': tf.shape(gt_boxes)[0],
          'image_info': info,
          'boxes': gt_boxes,
          'classes': gt_classes,
          'areas': tf.gather(data['groundtruth_area'], inds),
          'is_crowds':
              tf.cast(tf.gather(data['groundtruth_is_crowd'], inds), tf.int32),
      }
      groundtruths['source_id'] = utils.process_source_id(
          groundtruths['source_id'])
      groundtruths = utils.pad_groundtruths_to_fixed_size(
          groundtruths, self._max_num_instances)
      labels['groundtruths'] = groundtruths
    return image, labels
