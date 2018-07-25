# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Box predictor for object detectors.

Box predictors are classes that take a high level
image feature map as input and produce two predictions,
(1) a tensor encoding box locations, and
(2) a tensor encoding classes for each box.

These components are passed directly to loss functions
in our detection models.

These modules are separated from the main model since the same
few box predictor architectures are shared across many models.
"""
from abc import abstractmethod
import math
import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import static_shape

slim = tf.contrib.slim

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'
MASK_PREDICTIONS = 'mask_predictions'


class BoxPredictor(object):
  """BoxPredictor."""

  def __init__(self, is_training, num_classes):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
    """
    self._is_training = is_training
    self._num_classes = num_classes

  @property
  def num_classes(self):
    return self._num_classes

  def predict(self, image_features, num_predictions_per_location,
              scope=None, **params):
    """Computes encoded object locations and corresponding confidences.

    Takes a list of high level image feature maps as input and produces a list
    of box encodings and a list of class scores where each element in the output
    lists correspond to the feature maps in the input list.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      scope: Variable and Op scope name.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.

    Raises:
      ValueError: If length of `image_features` is not equal to length of
        `num_predictions_per_location`.
    """
    if len(image_features) != len(num_predictions_per_location):
      raise ValueError('image_feature and num_predictions_per_location must '
                       'be of same length, found: {} vs {}'.
                       format(len(image_features),
                              len(num_predictions_per_location)))
    if scope is not None:
      with tf.variable_scope(scope):
        return self._predict(image_features, num_predictions_per_location,
                             **params)
    return self._predict(image_features, num_predictions_per_location,
                         **params)

  # TODO(rathodv): num_predictions_per_location could be moved to constructor.
  # This is currently only used by ConvolutionalBoxPredictor.
  @abstractmethod
  def _predict(self, image_features, num_predictions_per_location, **params):
    """Implementations must override this method.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A list of float tensors. Each entry in the list
          corresponds to a feature map in the input `image_features` list. All
          tensors in the list have one of the two following shapes:
          a. [batch_size, num_anchors_i, q, code_size] representing the location
            of the objects, where q is 1 or the number of classes.
          b. [batch_size, num_anchors_i, code_size].
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
    """
    pass


class RfcnBoxPredictor(BoxPredictor):
  """RFCN Box Predictor.

  Applies a position sensitive ROI pooling on position sensitive feature maps to
  predict classes and refined locations. See https://arxiv.org/abs/1605.06409
  for details.

  This is used for the second stage of the RFCN meta architecture. Notice that
  locations are *not* shared across classes, thus for each anchor, a separate
  prediction is made for each class.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams_fn,
               num_spatial_bins,
               depth,
               crop_size,
               box_code_size):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to construct tf-slim arg_scope with
        hyperparameters for convolutional layers.
      num_spatial_bins: A list of two integers `[spatial_bins_y,
        spatial_bins_x]`.
      depth: Target depth to reduce the input feature maps to.
      crop_size: A list of two integers `[crop_height, crop_width]`.
      box_code_size: Size of encoding for each box.
    """
    super(RfcnBoxPredictor, self).__init__(is_training, num_classes)
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._num_spatial_bins = num_spatial_bins
    self._depth = depth
    self._crop_size = crop_size
    self._box_code_size = box_code_size

  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, num_predictions_per_location,
               proposal_boxes):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
        Currently, this must be set to [1], or an error will be raised.
      proposal_boxes: A float tensor of shape [batch_size, num_proposals,
        box_code_size].

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.

    Raises:
      ValueError: if num_predictions_per_location is not 1 or if
        len(image_features) is not 1.
    """
    if (len(num_predictions_per_location) != 1 or
        num_predictions_per_location[0] != 1):
      raise ValueError('Currently RfcnBoxPredictor only supports '
                       'predicting a single box per class per location.')
    if len(image_features) != 1:
      raise ValueError('length of `image_features` must be 1. Found {}'.
                       format(len(image_features)))
    image_feature = image_features[0]
    num_predictions_per_location = num_predictions_per_location[0]
    batch_size = tf.shape(proposal_boxes)[0]
    num_boxes = tf.shape(proposal_boxes)[1]
    def get_box_indices(proposals):
      proposals_shape = proposals.get_shape().as_list()
      if any(dim is None for dim in proposals_shape):
        proposals_shape = tf.shape(proposals)
      ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
      multiplier = tf.expand_dims(
          tf.range(start=0, limit=proposals_shape[0]), 1)
      return tf.reshape(ones_mat * multiplier, [-1])

    net = image_feature
    with slim.arg_scope(self._conv_hyperparams_fn()):
      net = slim.conv2d(net, self._depth, [1, 1], scope='reduce_depth')
      # Location predictions.
      location_feature_map_depth = (self._num_spatial_bins[0] *
                                    self._num_spatial_bins[1] *
                                    self.num_classes *
                                    self._box_code_size)
      location_feature_map = slim.conv2d(net, location_feature_map_depth,
                                         [1, 1], activation_fn=None,
                                         scope='refined_locations')
      box_encodings = ops.position_sensitive_crop_regions(
          location_feature_map,
          boxes=tf.reshape(proposal_boxes, [-1, self._box_code_size]),
          box_ind=get_box_indices(proposal_boxes),
          crop_size=self._crop_size,
          num_spatial_bins=self._num_spatial_bins,
          global_pool=True)
      box_encodings = tf.squeeze(box_encodings, squeeze_dims=[1, 2])
      box_encodings = tf.reshape(box_encodings,
                                 [batch_size * num_boxes, 1, self.num_classes,
                                  self._box_code_size])

      # Class predictions.
      total_classes = self.num_classes + 1  # Account for background class.
      class_feature_map_depth = (self._num_spatial_bins[0] *
                                 self._num_spatial_bins[1] *
                                 total_classes)
      class_feature_map = slim.conv2d(net, class_feature_map_depth, [1, 1],
                                      activation_fn=None,
                                      scope='class_predictions')
      class_predictions_with_background = ops.position_sensitive_crop_regions(
          class_feature_map,
          boxes=tf.reshape(proposal_boxes, [-1, self._box_code_size]),
          box_ind=get_box_indices(proposal_boxes),
          crop_size=self._crop_size,
          num_spatial_bins=self._num_spatial_bins,
          global_pool=True)
      class_predictions_with_background = tf.squeeze(
          class_predictions_with_background, squeeze_dims=[1, 2])
      class_predictions_with_background = tf.reshape(
          class_predictions_with_background,
          [batch_size * num_boxes, 1, total_classes])

    return {BOX_ENCODINGS: [box_encodings],
            CLASS_PREDICTIONS_WITH_BACKGROUND:
            [class_predictions_with_background]}


# TODO(rathodv): Change the implementation to return lists of predictions.
class MaskRCNNBoxPredictor(BoxPredictor):
  """Mask R-CNN Box Predictor.

  See Mask R-CNN: He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017).
  Mask R-CNN. arXiv preprint arXiv:1703.06870.

  This is used for the second stage of the Mask R-CNN detector where proposals
  cropped from an image are arranged along the batch dimension of the input
  image_features tensor. Notice that locations are *not* shared across classes,
  thus for each anchor, a separate prediction is made for each class.

  In addition to predicting boxes and classes, optionally this class allows
  predicting masks and/or keypoints inside detection boxes.

  Currently this box predictor makes per-class predictions; that is, each
  anchor makes a separate box prediction for each class.
  """

  def __init__(self,
               is_training,
               num_classes,
               fc_hyperparams_fn,
               use_dropout,
               dropout_keep_prob,
               box_code_size,
               conv_hyperparams_fn=None,
               predict_instance_masks=False,
               mask_height=14,
               mask_width=14,
               mask_prediction_num_conv_layers=2,
               mask_prediction_conv_depth=256,
               masks_are_class_agnostic=False,
               predict_keypoints=False,
               share_box_across_classes=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      fc_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for fully connected ops.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      box_code_size: Size of encoding for each box.
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      predict_instance_masks: Whether to predict object masks inside detection
        boxes.
      mask_height: Desired output mask height. The default value is 14.
      mask_width: Desired output mask width. The default value is 14.
      mask_prediction_num_conv_layers: Number of convolution layers applied to
        the image_features in mask prediction branch.
      mask_prediction_conv_depth: The depth for the first conv2d_transpose op
        applied to the image_features in the mask prediction branch. If set
        to 0, the depth of the convolution layers will be automatically chosen
        based on the number of object classes and the number of channels in the
        image features.
      masks_are_class_agnostic: Boolean determining if the mask-head is
        class-agnostic or not.
      predict_keypoints: Whether to predict keypoints insde detection boxes.
      share_box_across_classes: Whether to share boxes across classes rather
        than use a different box for each class.

    Raises:
      ValueError: If predict_instance_masks is true but conv_hyperparams is not
        set.
      ValueError: If predict_keypoints is true since it is not implemented yet.
      ValueError: If mask_prediction_num_conv_layers is smaller than two.
    """
    super(MaskRCNNBoxPredictor, self).__init__(is_training, num_classes)
    self._fc_hyperparams_fn = fc_hyperparams_fn
    self._use_dropout = use_dropout
    self._box_code_size = box_code_size
    self._dropout_keep_prob = dropout_keep_prob
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._predict_instance_masks = predict_instance_masks
    self._mask_height = mask_height
    self._mask_width = mask_width
    self._mask_prediction_num_conv_layers = mask_prediction_num_conv_layers
    self._mask_prediction_conv_depth = mask_prediction_conv_depth
    self._masks_are_class_agnostic = masks_are_class_agnostic
    self._predict_keypoints = predict_keypoints
    self._share_box_across_classes = share_box_across_classes
    if self._predict_keypoints:
      raise ValueError('Keypoint prediction is unimplemented.')
    if ((self._predict_instance_masks or self._predict_keypoints) and
        self._conv_hyperparams_fn is None):
      raise ValueError('`conv_hyperparams` must be provided when predicting '
                       'masks.')
    if self._mask_prediction_num_conv_layers < 2:
      raise ValueError(
          'Mask prediction should consist of at least 2 conv layers')

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def predicts_instance_masks(self):
    return self._predict_instance_masks

  def _predict_boxes_and_classes(self, image_features):
    """Predicts boxes and class scores.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, 1, num_classes, code_size] representing the location of the
        objects.
      class_predictions_with_background: A float tensor of shape
        [batch_size, 1, num_classes + 1] representing the class predictions for
        the proposals.
    """
    spatial_averaged_image_features = tf.reduce_mean(image_features, [1, 2],
                                                     keep_dims=True,
                                                     name='AvgPool')
    flattened_image_features = slim.flatten(spatial_averaged_image_features)
    if self._use_dropout:
      flattened_image_features = slim.dropout(flattened_image_features,
                                              keep_prob=self._dropout_keep_prob,
                                              is_training=self._is_training)
    number_of_boxes = 1
    if not self._share_box_across_classes:
      number_of_boxes = self._num_classes

    with slim.arg_scope(self._fc_hyperparams_fn()):
      box_encodings = slim.fully_connected(
          flattened_image_features,
          number_of_boxes * self._box_code_size,
          activation_fn=None,
          scope='BoxEncodingPredictor')
      class_predictions_with_background = slim.fully_connected(
          flattened_image_features,
          self._num_classes + 1,
          activation_fn=None,
          scope='ClassPredictor')
    box_encodings = tf.reshape(
        box_encodings, [-1, 1, number_of_boxes, self._box_code_size])
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background, [-1, 1, self._num_classes + 1])
    return box_encodings, class_predictions_with_background

  def _get_mask_predictor_conv_depth(self, num_feature_channels, num_classes,
                                     class_weight=3.0, feature_weight=2.0):
    """Computes the depth of the mask predictor convolutions.

    Computes the depth of the mask predictor convolutions given feature channels
    and number of classes by performing a weighted average of the two in
    log space to compute the number of convolution channels. The weights that
    are used for computing the weighted average do not need to sum to 1.

    Args:
      num_feature_channels: An integer containing the number of feature
        channels.
      num_classes: An integer containing the number of classes.
      class_weight: Class weight used in computing the weighted average.
      feature_weight: Feature weight used in computing the weighted average.

    Returns:
      An integer containing the number of convolution channels used by mask
        predictor.
    """
    num_feature_channels_log = math.log(float(num_feature_channels), 2.0)
    num_classes_log = math.log(float(num_classes), 2.0)
    weighted_num_feature_channels_log = (
        num_feature_channels_log * feature_weight)
    weighted_num_classes_log = num_classes_log * class_weight
    total_weight = feature_weight + class_weight
    num_conv_channels_log = round(
        (weighted_num_feature_channels_log + weighted_num_classes_log) /
        total_weight)
    return int(math.pow(2.0, num_conv_channels_log))

  def _predict_masks(self, image_features):
    """Performs mask prediction.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      instance_masks: A float tensor of shape
          [batch_size, 1, num_classes, image_height, image_width].
    """
    num_conv_channels = self._mask_prediction_conv_depth
    if num_conv_channels == 0:
      num_feature_channels = image_features.get_shape().as_list()[3]
      num_conv_channels = self._get_mask_predictor_conv_depth(
          num_feature_channels, self.num_classes)
    with slim.arg_scope(self._conv_hyperparams_fn()):
      upsampled_features = tf.image.resize_bilinear(
          image_features,
          [self._mask_height, self._mask_width],
          align_corners=True)
      for _ in range(self._mask_prediction_num_conv_layers - 1):
        upsampled_features = slim.conv2d(
            upsampled_features,
            num_outputs=num_conv_channels,
            kernel_size=[3, 3])
      num_masks = 1 if self._masks_are_class_agnostic else self.num_classes
      mask_predictions = slim.conv2d(upsampled_features,
                                     num_outputs=num_masks,
                                     activation_fn=None,
                                     kernel_size=[3, 3])
      return tf.expand_dims(
          tf.transpose(mask_predictions, perm=[0, 3, 1, 2]),
          axis=1,
          name='MaskPredictor')

  def _predict(self, image_features, num_predictions_per_location,
               predict_boxes_and_classes=True, predict_auxiliary_outputs=False):
    """Optionally computes encoded object locations, confidences, and masks.

    Flattens image_features and applies fully connected ops (with no
    non-linearity) to predict box encodings and class predictions.  In this
    setting, anchors are not spatially arranged in any way and are assumed to
    have been folded into the batch dimension.  Thus we output 1 for the
    anchors dimension.

    Also optionally predicts instance masks.
    The mask prediction head is based on the Mask RCNN paper with the following
    modifications: We replace the deconvolution layer with a bilinear resize
    and a convolution.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location: A list of integers representing the number
        of box predictions to be made per spatial location for each feature map.
        Currently, this must be set to [1], or an error will be raised.
      predict_boxes_and_classes: If true, the function will perform box
        refinement and classification.
      predict_auxiliary_outputs: If true, the function will perform other
        predictions such as mask, keypoint, boundaries, etc. if any.

    Returns:
      A dictionary containing the following tensors.
        box_encodings: A float tensor of shape
          [batch_size, 1, num_classes, code_size] representing the
          location of the objects.
        class_predictions_with_background: A float tensor of shape
          [batch_size, 1, num_classes + 1] representing the class
          predictions for the proposals.
      If predict_masks is True the dictionary also contains:
        instance_masks: A float tensor of shape
          [batch_size, 1, num_classes, image_height, image_width]
      If predict_keypoints is True the dictionary also contains:
        keypoints: [batch_size, 1, num_keypoints, 2]

    Raises:
      ValueError: If num_predictions_per_location is not 1 or if both
        predict_boxes_and_classes and predict_auxiliary_outputs are false or if
        len(image_features) is not 1.
    """
    if (len(num_predictions_per_location) != 1 or
        num_predictions_per_location[0] != 1):
      raise ValueError('Currently FullyConnectedBoxPredictor only supports '
                       'predicting a single box per class per location.')
    if not predict_boxes_and_classes and not predict_auxiliary_outputs:
      raise ValueError('Should perform at least one prediction.')
    if len(image_features) != 1:
      raise ValueError('length of `image_features` must be 1. Found {}'.
                       format(len(image_features)))
    image_feature = image_features[0]
    num_predictions_per_location = num_predictions_per_location[0]
    predictions_dict = {}

    if predict_boxes_and_classes:
      (box_encodings, class_predictions_with_background
      ) = self._predict_boxes_and_classes(image_feature)
      predictions_dict[BOX_ENCODINGS] = box_encodings
      predictions_dict[
          CLASS_PREDICTIONS_WITH_BACKGROUND] = class_predictions_with_background

    if self._predict_instance_masks and predict_auxiliary_outputs:
      predictions_dict[MASK_PREDICTIONS] = self._predict_masks(image_feature)

    return predictions_dict


class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class ConvolutionalBoxPredictor(BoxPredictor):
  """Convolutional Box Predictor.

  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.

  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams_fn,
               min_depth,
               max_depth,
               num_layers_before_predictor,
               use_dropout,
               dropout_keep_prob,
               kernel_size,
               box_code_size,
               apply_sigmoid_to_scores=False,
               class_prediction_bias_init=0.0,
               use_depthwise=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      min_depth: Minimum feature depth prior to predicting box encodings
        and class predictions.
      max_depth: Maximum feature depth prior to predicting box encodings
        and class predictions. If max_depth is set to 0, no additional
        feature map will be inserted before location and class predictions.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      use_dropout: Option to use dropout for class prediction or not.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      box_code_size: Size of encoding for each box.
      apply_sigmoid_to_scores: if True, apply the sigmoid on the output
        class_predictions.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
    if min_depth > max_depth:
      raise ValueError('min_depth should be less than or equal to max_depth')
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._use_dropout = use_dropout
    self._kernel_size = kernel_size
    self._box_code_size = box_code_size
    self._dropout_keep_prob = dropout_keep_prob
    self._apply_sigmoid_to_scores = apply_sigmoid_to_scores
    self._class_prediction_bias_init = class_prediction_bias_init
    self._use_depthwise = use_depthwise

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """
    box_encodings_list = []
    class_predictions_list = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]

    for (image_feature,
         num_predictions_per_location, box_predictor_scope) in zip(
             image_features, num_predictions_per_location_list,
             box_predictor_scopes):
      with box_predictor_scope:
        # Add a slot for the background class.
        num_class_slots = self.num_classes + 1
        net = image_feature
        with slim.arg_scope(self._conv_hyperparams_fn()), \
             slim.arg_scope([slim.dropout], is_training=self._is_training):
          # Add additional conv layers before the class predictor.
          features_depth = static_shape.get_depth(image_feature.get_shape())
          depth = max(min(features_depth, self._max_depth), self._min_depth)
          tf.logging.info('depth of additional conv before box predictor: {}'.
                          format(depth))
          if depth > 0 and self._num_layers_before_predictor > 0:
            for i in range(self._num_layers_before_predictor):
              net = slim.conv2d(
                  net, depth, [1, 1], scope='Conv2d_%d_1x1_%d' % (i, depth))
          with slim.arg_scope([slim.conv2d], activation_fn=None,
                              normalizer_fn=None, normalizer_params=None):
            if self._use_depthwise:
              box_encodings = slim.separable_conv2d(
                  net, None, [self._kernel_size, self._kernel_size],
                  padding='SAME', depth_multiplier=1, stride=1,
                  rate=1, scope='BoxEncodingPredictor_depthwise')
              box_encodings = slim.conv2d(
                  box_encodings,
                  num_predictions_per_location * self._box_code_size, [1, 1],
                  scope='BoxEncodingPredictor')
            else:
              box_encodings = slim.conv2d(
                  net, num_predictions_per_location * self._box_code_size,
                  [self._kernel_size, self._kernel_size],
                  scope='BoxEncodingPredictor')
            if self._use_dropout:
              net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
            if self._use_depthwise:
              class_predictions_with_background = slim.separable_conv2d(
                  net, None, [self._kernel_size, self._kernel_size],
                  padding='SAME', depth_multiplier=1, stride=1,
                  rate=1, scope='ClassPredictor_depthwise')
              class_predictions_with_background = slim.conv2d(
                  class_predictions_with_background,
                  num_predictions_per_location * num_class_slots,
                  [1, 1], scope='ClassPredictor')
            else:
              class_predictions_with_background = slim.conv2d(
                  net, num_predictions_per_location * num_class_slots,
                  [self._kernel_size, self._kernel_size],
                  scope='ClassPredictor',
                  biases_initializer=tf.constant_initializer(
                      self._class_prediction_bias_init))
            if self._apply_sigmoid_to_scores:
              class_predictions_with_background = tf.sigmoid(
                  class_predictions_with_background)

        combined_feature_map_shape = (shape_utils.
                                      combined_static_and_dynamic_shape(
                                          image_feature))
        box_encodings = tf.reshape(
            box_encodings, tf.stack([combined_feature_map_shape[0],
                                     combined_feature_map_shape[1] *
                                     combined_feature_map_shape[2] *
                                     num_predictions_per_location,
                                     1, self._box_code_size]))
        box_encodings_list.append(box_encodings)
        class_predictions_with_background = tf.reshape(
            class_predictions_with_background,
            tf.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))
        class_predictions_list.append(class_predictions_with_background)
    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }


# TODO(rathodv): Replace with slim.arg_scope_func_key once its available
# externally.
def _arg_scope_func_key(op):
  """Returns a key that can be used to index arg_scope dictionary."""
  return getattr(op, '_key_op', str(op))


# TODO(rathodv): Merge the implementation with ConvolutionalBoxPredictor above
# since they are very similar.
class WeightSharedConvolutionalBoxPredictor(BoxPredictor):
  """Convolutional Box Predictor with weight sharing.

  Defines the box predictor as defined in
  https://arxiv.org/abs/1708.02002. This class differs from
  ConvolutionalBoxPredictor in that it shares weights and biases while
  predicting from different feature maps. However, batch_norm parameters are not
  shared because the statistics of the activations vary among the different
  feature maps.

  Also note that separate multi-layer towers are constructed for the box
  encoding and class predictors respectively.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams_fn,
               depth,
               num_layers_before_predictor,
               box_code_size,
               kernel_size=3,
               class_prediction_bias_init=0.0,
               use_dropout=False,
               dropout_keep_prob=0.8,
               share_prediction_tower=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      depth: depth of conv layers.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_dropout: Whether to apply dropout to class prediction head.
      dropout_keep_prob: Probability of keeping activiations.
      share_prediction_tower: Whether to share the multi-layer tower between box
        prediction and class prediction heads.
    """
    super(WeightSharedConvolutionalBoxPredictor, self).__init__(is_training,
                                                                num_classes)
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._depth = depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._class_prediction_bias_init = class_prediction_bias_init
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._share_prediction_tower = share_prediction_tower

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels] containing features for a batch of images. Note that
        when not all tensors in the list have the same number of channels, an
        additional projection layer will be added on top the tensor to generate
        feature map with number of channels consitent with the majority.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map. Note that all values must be the same since the weights are
        shared.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, code_size] representing the location of
        the objects. Each entry in the list corresponds to a feature map in the
        input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.


    Raises:
      ValueError: If the image feature maps do not have the same number of
        channels or if the num predictions per locations is differs between the
        feature maps.
    """
    if len(set(num_predictions_per_location_list)) > 1:
      raise ValueError('num predictions per location must be same for all'
                       'feature maps, found: {}'.format(
                           num_predictions_per_location_list))
    feature_channels = [
        image_feature.shape[3].value for image_feature in image_features
    ]
    has_different_feature_channels = len(set(feature_channels)) > 1
    if has_different_feature_channels:
      inserted_layer_counter = 0
      target_channel = max(set(feature_channels), key=feature_channels.count)
      tf.logging.info('Not all feature maps have the same number of '
                      'channels, found: {}, addition project layers '
                      'to bring all feature maps to uniform channels '
                      'of {}'.format(feature_channels, target_channel))
    box_encodings_list = []
    class_predictions_list = []
    num_class_slots = self.num_classes + 1
    for feature_index, (image_feature,
                        num_predictions_per_location) in enumerate(
                            zip(image_features,
                                num_predictions_per_location_list)):
      # Add a slot for the background class.
      with tf.variable_scope('WeightSharedConvolutionalBoxPredictor',
                             reuse=tf.AUTO_REUSE):
        with slim.arg_scope(self._conv_hyperparams_fn()) as sc:
          apply_batch_norm = _arg_scope_func_key(slim.batch_norm) in sc
          # Insert an additional projection layer if necessary.
          if (has_different_feature_channels and
              image_feature.shape[3].value != target_channel):
            image_feature = slim.conv2d(
                image_feature,
                target_channel, [1, 1],
                stride=1,
                padding='SAME',
                activation_fn=None,
                normalizer_fn=(tf.identity if apply_batch_norm else None),
                scope='ProjectionLayer/conv2d_{}'.format(
                    inserted_layer_counter))
            if apply_batch_norm:
              image_feature = slim.batch_norm(
                  image_feature,
                  scope='ProjectionLayer/conv2d_{}/BatchNorm'.format(
                      inserted_layer_counter))
            inserted_layer_counter += 1
          box_encodings_net = image_feature
          class_predictions_net = image_feature
          for i in range(self._num_layers_before_predictor):
            box_prediction_tower_prefix = (
                'PredictionTower' if self._share_prediction_tower
                else 'BoxPredictionTower')
            box_encodings_net = slim.conv2d(
                box_encodings_net,
                self._depth,
                [self._kernel_size, self._kernel_size],
                stride=1,
                padding='SAME',
                activation_fn=None,
                normalizer_fn=(tf.identity if apply_batch_norm else None),
                scope='{}/conv2d_{}'.format(box_prediction_tower_prefix, i))
            if apply_batch_norm:
              box_encodings_net = slim.batch_norm(
                  box_encodings_net,
                  scope='{}/conv2d_{}/BatchNorm/feature_{}'.
                  format(box_prediction_tower_prefix, i, feature_index))
            box_encodings_net = tf.nn.relu6(box_encodings_net)
          box_encodings = slim.conv2d(
              box_encodings_net,
              num_predictions_per_location * self._box_code_size,
              [self._kernel_size, self._kernel_size],
              activation_fn=None, stride=1, padding='SAME',
              normalizer_fn=None,
              scope='BoxPredictor')

          if self._share_prediction_tower:
            class_predictions_net = box_encodings_net
          else:
            for i in range(self._num_layers_before_predictor):
              class_predictions_net = slim.conv2d(
                  class_predictions_net,
                  self._depth,
                  [self._kernel_size, self._kernel_size],
                  stride=1,
                  padding='SAME',
                  activation_fn=None,
                  normalizer_fn=(tf.identity if apply_batch_norm else None),
                  scope='ClassPredictionTower/conv2d_{}'.format(i))
              if apply_batch_norm:
                class_predictions_net = slim.batch_norm(
                    class_predictions_net,
                    scope='ClassPredictionTower/conv2d_{}/BatchNorm/feature_{}'
                    .format(i, feature_index))
              class_predictions_net = tf.nn.relu6(class_predictions_net)
          if self._use_dropout:
            class_predictions_net = slim.dropout(
                class_predictions_net, keep_prob=self._dropout_keep_prob)
          class_predictions_with_background = slim.conv2d(
              class_predictions_net,
              num_predictions_per_location * num_class_slots,
              [self._kernel_size, self._kernel_size],
              activation_fn=None, stride=1, padding='SAME',
              normalizer_fn=None,
              biases_initializer=tf.constant_initializer(
                  self._class_prediction_bias_init),
              scope='ClassPredictor')

          combined_feature_map_shape = (shape_utils.
                                        combined_static_and_dynamic_shape(
                                            image_feature))
          box_encodings = tf.reshape(
              box_encodings, tf.stack([combined_feature_map_shape[0],
                                       combined_feature_map_shape[1] *
                                       combined_feature_map_shape[2] *
                                       num_predictions_per_location,
                                       self._box_code_size]))
          box_encodings_list.append(box_encodings)
          class_predictions_with_background = tf.reshape(
              class_predictions_with_background,
              tf.stack([combined_feature_map_shape[0],
                        combined_feature_map_shape[1] *
                        combined_feature_map_shape[2] *
                        num_predictions_per_location,
                        num_class_slots]))
          class_predictions_list.append(class_predictions_with_background)
    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }
