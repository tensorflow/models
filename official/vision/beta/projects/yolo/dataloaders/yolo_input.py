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

""" Detection Data parser and processing for YOLO."""
import tensorflow as tf
import numpy as np
from official.vision.beta.projects.yolo.ops import preprocessing_ops
from official.vision.beta.projects.yolo.ops import box_ops as box_utils
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.dataloaders import parser, utils


def _coco91_to_80(classif, box, areas, iscrowds):
  """Function used to reduce COCO 91 to COCO 80, or to convert from the 2017 
  foramt to the 2014 format"""
  # Vector where index i coralates to the class at index[i].
  x = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
      23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
      44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
      63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
      86, 87, 88, 89, 90
  ]
  no = tf.expand_dims(tf.convert_to_tensor(x), axis=0)

  # Resahpe the classes to in order to build a class mask.
  ce = tf.expand_dims(classif, axis=-1)

  # One hot the classificiations to match the 80 class format.
  ind = ce == tf.cast(no, ce.dtype)

  # Select the max values.
  co = tf.reshape(tf.math.argmax(tf.cast(ind, tf.float32), axis=-1), [-1])
  ind = tf.where(tf.reduce_any(ind, axis=-1))

  # Gather the valuable instances.
  classif = tf.gather_nd(co, ind)
  box = tf.gather_nd(box, ind)
  areas = tf.gather_nd(areas, ind)
  iscrowds = tf.gather_nd(iscrowds, ind)

  # Restate the number of viable detections, ideally it should be the same.
  num_detections = tf.shape(classif)[0]
  return classif, box, areas, iscrowds, num_detections


class Parser(parser.Parser):
  """Parse the dataset in to the YOLO model format. """

  def __init__(
      self,
      output_size,
      masks,
      anchors,
      strides,
      anchor_free_limits=None,
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
      coco91to80=False,
      darknet=False,
      use_tie_breaker=True,
      dtype='float32',
      seed=None,
  ):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `List` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      masks: `Dict[List[int]]` of values indicating the indexes in the 
        list of anchor boxes to use an each prediction level between min_level 
        and max_level. each level must have a list of indexes.  
      anchors: `List[List[Union[int, float]]]` values for each anchor box.
      strides: `Dict[int]` for how much the model scales down the images at the
        largest level.
      anchor_free_limits: `List` the box sizes that will be allowed at each FPN 
        level as is done in the FCOS and YOLOX paper for anchor free box 
        assignment. Anchor free will perform worse than Anchor based, but only 
        slightly.
      max_num_instances: `int` for the number of boxes to compute loss on.
      area_thresh: `float` for the minimum area of a box to allow to pass 
        through for optimization.
      aug_rand_hue: `float` indicating the maximum scaling value for 
        hue. saturation will be scaled between 1 - value and 1 + value.
      aug_rand_saturation: `float` indicating the maximum scaling value for 
        saturation. saturation will be scaled between 1/value and value.
      aug_rand_brightness: `float` indicating the maximum scaling value for 
        brightness. brightness will be scaled between 1/value and value.
      letter_box: `boolean` indicating whether upon start of the datapipeline 
        regardless of the preprocessing ops that are used, the aspect ratio of 
        the images should be preserved.  
      random_pad: `bool` indiccating wether to use padding to apply random 
        translation true for darknet yolo false for scaled yolo.
      random_flip: `boolean` indicating whether or not to randomly flip the 
        image horizontally. 
      jitter: `float` for the maximum change in aspect ratio expected in 
        each preprocessing step.
      aug_scale_min: `float` indicating the minimum scaling value for image 
        scale jitter. 
      aug_scale_max: `float` indicating the maximum scaling value for image 
        scale jitter.
      aug_rand_translate: `float` ranging from 0 to 1 indicating the maximum 
        amount to randomly translate an image.
      aug_rand_perspective: `float` ranging from 0.000 to 0.001 indicating 
        how much to prespective warp the image.
      aug_rand_angle: `float` indicating the maximum angle value for 
        angle. angle will be changes between 0 and value.
      anchor_t: `float` indicating the threshold over which an anchor will be 
        considered for prediction, at zero, all the anchors will be used and at
        1.0 only the best will be used. for anchor thresholds larger than 1.0 
        we stop using the IOU for anchor comparison and resort directly to 
        comparing the width and height, this is used for the scaled models.  
      scale_xy: dictionary `float` values inidcating how far each pixel can see 
        outside of its containment of 1.0. a value of 1.2 indicates there is a 
        20% extended radius around each pixel that this specific pixel can 
        predict values for a center at. the center can range from 0 - value/2 
        to 1 + value/2, this value is set in the yolo filter, and resused here. 
        there should be one value for scale_xy for each level from min_level to 
        max_level.
      best_match_only: `boolean` indicating how boxes are selected for 
        optimization.
      coco91to80: `bool` for wether to convert coco91 to coco80 to minimize 
        model parameters.      
      darknet: `boolean` indicating which data pipeline to use. Setting to True 
        swaps the pipeline to output images realtive to Yolov4 and older. 
      use_tie_breaker: `boolean` indicating whether to use the anchor threshold 
        value.
      dtype: `str` indicating the output datatype of the datapipeline selecting 
        from {"float32", "float16", "bfloat16"}.
      seed: `int` the seed for random number generation. 
    """
    for key in masks.keys():
      # Assert that the width and height is viable
      assert output_size[1] % strides[str(key)] == 0
      assert output_size[0] % strides[str(key)] == 0

    # scale of each FPN level
    self._strides = strides

    # Set the width and height properly and base init:
    self._coco91to80 = coco91to80
    self._image_w = output_size[1]
    self._image_h = output_size[0]

    # Set the anchor boxes and masks for each scale
    self._anchors = anchors
    self._anchor_free_limits = anchor_free_limits
    self._masks = {
        key: tf.convert_to_tensor(value) for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker
    self._best_match_only = best_match_only
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
    self._scale_xy = scale_xy
    self._anchor_t = anchor_t
    self._darknet = darknet
    self._area_thresh = area_thresh

    keys = list(self._masks.keys())

    if self._anchor_free_limits is not None:
      maxim = 2000
      self._scale_up = {key: maxim // self._max_num_instances for key in keys}
      self._anchor_t = -0.01
    elif not self._darknet:
      self._scale_up = {key: 6 - i for i, key in enumerate(keys)}
    else:
      self._scale_up = {key: 1 for key in keys}

    self._seed = seed

    # Set the data type based on input string
    self._dtype = dtype

  def _get_identity_info(self, image):
    """Get an identity image op to pad all info vectors, this is used because 
    graph compilation if there are a variable number of info objects in a list.
    """
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
    if (aug_scale_min != 1.0 or aug_scale_max != 1.0):
      crop_only = True
      # jitter gives you only one info object, resize and crop gives you one,
      # if crop only then there can be 1 form jitter and 1 from crop
      reps = 1
    else:
      crop_only = False
      reps = 0
    infos = []
    image, info_a, _ = preprocessing_ops.resize_and_jitter_image(
        image,
        shape,
        letter_box=letter_box,
        jitter=jitter,
        crop_only=crop_only,
        random_pad=random_pad,
        seed=self._seed,
    )
    infos.extend(info_a)
    stale_a = self._get_identity_info(image)
    for _ in range(reps):
      infos.append(stale_a)
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

  def reorg91to80(self, data):
    """Function used to reduce COCO 91 to COCO 80, or to convert from the 2017 
    foramt to the 2014 format"""
    if self._coco91to80:
      (data['groundtruth_classes'], data['groundtruth_boxes'],
       data['groundtruth_area'], data['groundtruth_is_crowd'],
       _) = _coco91_to_80(data['groundtruth_classes'],
                          data['groundtruth_boxes'], data['groundtruth_area'],
                          data['groundtruth_is_crowd'])
    return data

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    # Down size coco 91 to coco 80 if the option is selected.
    data = self.reorg91to80(data)

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
      boxes, inds = preprocessing_ops.apply_infos(
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
      inds = tf.cast(tf.range(0, tf.shape(boxes)[0]), tf.int64)
      info = self._get_identity_info(image)

    # Apply scaling to the hue saturation and brightness of an image.
    image = tf.cast(image, dtype=self._dtype)
    image = image / 255
    image = preprocessing_ops.image_rand_hsv(
        image,
        self._aug_rand_hue,
        self._aug_rand_saturation,
        self._aug_rand_brightness,
        seed=self._seed,
        darknet=self._darknet)

    # Cast the image to the selcted datatype.
    image, labels = self._build_label(
        image,
        boxes,
        classes,
        self._image_w,
        self._image_h,
        info,
        inds,
        data,
        is_training=True)
    return image, labels

  def _parse_eval_data(self, data):
    # Down size coco 91 to coco 80 if the option is selected.
    data = self.reorg91to80(data)

    # Get the image shape constants and cast the image to the selcted datatype.
    image = tf.cast(data['image'], dtype=self._dtype)
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    height, width = self._image_h, self._image_w
    image, infos, _ = preprocessing_ops.resize_and_jitter_image(
        image, [height, width],
        letter_box=self._letter_box,
        random_pad=False,
        shiftx=0.5,
        shifty=0.5,
        jitter=0.0)

    # Clip and clean boxes.
    image = image / 255
    boxes, inds = preprocessing_ops.apply_infos(
        boxes, infos, shuffle_boxes=False, area_thresh=0.0, augment=True)
    classes = tf.gather(classes, inds)
    info = infos[-1]

    image, labels = self._build_label(
        image,
        boxes,
        classes,
        width,
        height,
        info,
        inds,
        data,
        is_training=False)
    return image, labels

  def set_shape(self, values, pad_axis=0, pad_value=0, inds=None, scale=1):
    if inds is not None:
      values = tf.gather(values, inds)
    vshape = values.get_shape().as_list()

    if pad_value is not None:
      values = preprocessing_ops.pad_max_instances(
          values,
          self._max_num_instances,
          pad_axis=pad_axis,
          pad_value=pad_value)

    vshape[pad_axis] = self._max_num_instances * scale
    values.set_shape(vshape)
    return values

  def _build_grid(self, raw_true, width, height, use_tie_breaker=False):
    '''Private function for building the full scale object and class grid.'''
    indexes = {}
    updates = {}
    true_grids = {}

    if self._anchor_free_limits is not None:
      self._anchor_free_limits = [0.0] + self._anchor_free_limits + [np.inf]

    # for each prediction path generate a properly scaled output prediction map
    for i, key in enumerate(self._masks.keys()):
      if self._anchor_free_limits is not None:
        fpn_limits = self._anchor_free_limits[i:i + 2]
      else:
        fpn_limits = None

      # build the actual grid as well and the list of boxes and classes AND
      # their index in the prediction grid
      scale_xy = self._scale_xy[key] if not self._darknet else 1
      (indexes[key], updates[key],
       true_grids[key]) = preprocessing_ops.build_grided_gt_ind(
           raw_true,
           self._masks[key],
           width // self._strides[str(key)],
           height // self._strides[str(key)],
           raw_true['bbox'].dtype,
           scale_xy,
           self._scale_up[key],
           use_tie_breaker,
           self._strides[str(key)],
           fpn_limits=fpn_limits)

      # set/fix the shapes
      indexes[key] = self.set_shape(indexes[key], -2, None, None,
                                    self._scale_up[key])
      updates[key] = self.set_shape(updates[key], -2, None, None,
                                    self._scale_up[key])

      # add all the values to the final dictionary
      updates[key] = tf.cast(updates[key], dtype=self._dtype)
    return indexes, updates, true_grids

  def _build_label(self,
                   image,
                   gt_boxes,
                   gt_classes,
                   width,
                   height,
                   info,
                   inds,
                   data,
                   is_training=True):
    """Label construction for both the train and eval data. """
    # Set the image shape.
    imshape = image.get_shape().as_list()
    imshape[-1] = 3
    image.set_shape(imshape)

    # Get the best anchors.
    boxes = box_utils.yxyx_to_xcycwh(gt_boxes)
    best_anchors, ious = preprocessing_ops.get_best_anchor(
        boxes,
        self._anchors,
        width=width,
        height=height,
        iou_thresh=self._anchor_t,
        best_match_only=self._best_match_only)

    # Set/fix the boxes shape.
    boxes = self.set_shape(boxes, pad_axis=0, pad_value=0)
    classes = self.set_shape(gt_classes, pad_axis=0, pad_value=-1)
    best_anchors = self.set_shape(best_anchors, pad_axis=0, pad_value=-1)
    ious = self.set_shape(ious, pad_axis=0, pad_value=0)
    area = self.set_shape(
        data['groundtruth_area'], pad_axis=0, pad_value=0, inds=inds)
    is_crowd = self.set_shape(
        data['groundtruth_is_crowd'], pad_axis=0, pad_value=0, inds=inds)

    # Build the dictionary set.
    labels = {
        'source_id': utils.process_source_id(data['source_id']),
        'bbox': tf.cast(boxes, dtype=self._dtype),
        'classes': tf.cast(classes, dtype=self._dtype),
        'best_anchors': tf.cast(best_anchors, dtype=self._dtype),
        'best_iou_match': ious,
    }

    # Build the grid formatted for loss computation in model output format.
    labels['inds'], labels['upds'], labels['true_conf'] = self._build_grid(
        labels, width, height, use_tie_breaker=self._use_tie_breaker)

    # Update the labels dictionary.
    labels['bbox'] = box_utils.xcycwh_to_yxyx(labels['bbox'])

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
          'areas': area,
          'is_crowds': tf.cast(is_crowd, tf.int32),
      }
      groundtruths['source_id'] = utils.process_source_id(
          groundtruths['source_id'])
      groundtruths = utils.pad_groundtruths_to_fixed_size(
          groundtruths, self._max_num_instances)
      labels['groundtruths'] = groundtruths
    return image, labels
