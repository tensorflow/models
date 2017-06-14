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
import tensorflow as tf
from object_detection.utils import ops
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

  def predict(self, image_features, num_predictions_per_location, scope,
              **params):
    """Computes encoded object locations and corresponding confidences.

    Takes a high level image feature map as input and produce two predictions,
    (1) a tensor encoding box locations, and
    (2) a tensor encoding class scores for each corresponding box.
    In this interface, we only assume that two tensors are returned as output
    and do not assume anything about their shapes.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.
      scope: Variable and Op scope name.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A float tensor of shape
          [batch_size, num_anchors, q, code_size] representing the location of
          the objects, where q is 1 or the number of classes.
        class_predictions_with_background: A float tensor of shape
          [batch_size, num_anchors, num_classes + 1] representing the class
          predictions for the proposals.
    """
    with tf.variable_scope(scope):
      return self._predict(image_features, num_predictions_per_location,
                           **params)

  # TODO: num_predictions_per_location could be moved to constructor.
  # This is currently only used by ConvolutionalBoxPredictor.
  @abstractmethod
  def _predict(self, image_features, num_predictions_per_location, **params):
    """Implementations must override this method.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.
      **params: Additional keyword arguments for specific implementations of
              BoxPredictor.

    Returns:
      A dictionary containing at least the following tensors.
        box_encodings: A float tensor of shape
          [batch_size, num_anchors, q, code_size] representing the location of
          the objects, where q is 1 or the number of classes.
        class_predictions_with_background: A float tensor of shape
          [batch_size, num_anchors, num_classes + 1] representing the class
          predictions for the proposals.
    """
    pass


class RfcnBoxPredictor(BoxPredictor):
  """RFCN Box Predictor.

  Applies a position sensitve ROI pooling on position sensitive feature maps to
  predict classes and refined locations. See https://arxiv.org/abs/1605.06409
  for details.

  This is used for the second stage of the RFCN meta architecture. Notice that
  locations are *not* shared across classes, thus for each anchor, a separate
  prediction is made for each class.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams,
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
      conv_hyperparams: Slim arg_scope with hyperparameters for conolutional
        layers.
      num_spatial_bins: A list of two integers `[spatial_bins_y,
        spatial_bins_x]`.
      depth: Target depth to reduce the input feature maps to.
      crop_size: A list of two integers `[crop_height, crop_width]`.
      box_code_size: Size of encoding for each box.
    """
    super(RfcnBoxPredictor, self).__init__(is_training, num_classes)
    self._conv_hyperparams = conv_hyperparams
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
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.
        Currently, this must be set to 1, or an error will be raised.
      proposal_boxes: A float tensor of shape [batch_size, num_proposals,
        box_code_size].

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, 1, num_classes, code_size] representing the
        location of the objects.
      class_predictions_with_background: A float tensor of shape
        [batch_size, 1, num_classes + 1] representing the class
        predictions for the proposals.
    Raises:
      ValueError: if num_predictions_per_location is not 1.
    """
    if num_predictions_per_location != 1:
      raise ValueError('Currently RfcnBoxPredictor only supports '
                       'predicting a single box per class per location.')

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

    net = image_features
    with slim.arg_scope(self._conv_hyperparams):
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

    return {BOX_ENCODINGS: box_encodings,
            CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background}


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
               fc_hyperparams,
               use_dropout,
               dropout_keep_prob,
               box_code_size,
               conv_hyperparams=None,
               predict_instance_masks=False,
               mask_prediction_conv_depth=256,
               predict_keypoints=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      fc_hyperparams: Slim arg_scope with hyperparameters for fully
        connected ops.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      box_code_size: Size of encoding for each box.
      conv_hyperparams: Slim arg_scope with hyperparameters for convolution
        ops.
      predict_instance_masks: Whether to predict object masks inside detection
        boxes.
      mask_prediction_conv_depth: The depth for the first conv2d_transpose op
        applied to the image_features in the mask prediciton branch.
      predict_keypoints: Whether to predict keypoints insde detection boxes.


    Raises:
      ValueError: If predict_instance_masks or predict_keypoints is true.
    """
    super(MaskRCNNBoxPredictor, self).__init__(is_training, num_classes)
    self._fc_hyperparams = fc_hyperparams
    self._use_dropout = use_dropout
    self._box_code_size = box_code_size
    self._dropout_keep_prob = dropout_keep_prob
    self._conv_hyperparams = conv_hyperparams
    self._predict_instance_masks = predict_instance_masks
    self._mask_prediction_conv_depth = mask_prediction_conv_depth
    self._predict_keypoints = predict_keypoints
    if self._predict_keypoints:
      raise ValueError('Keypoint prediction is unimplemented.')
    if ((self._predict_instance_masks or self._predict_keypoints) and
        self._conv_hyperparams is None):
      raise ValueError('`conv_hyperparams` must be provided when predicting '
                       'masks.')

  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, num_predictions_per_location):
    """Computes encoded object locations and corresponding confidences.

    Flattens image_features and applies fully connected ops (with no
    non-linearity) to predict box encodings and class predictions.  In this
    setting, anchors are not spatially arranged in any way and are assumed to
    have been folded into the batch dimension.  Thus we output 1 for the
    anchors dimension.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.
        Currently, this must be set to 1, or an error will be raised.

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
      ValueError: if num_predictions_per_location is not 1.
    """
    if num_predictions_per_location != 1:
      raise ValueError('Currently FullyConnectedBoxPredictor only supports '
                       'predicting a single box per class per location.')
    spatial_averaged_image_features = tf.reduce_mean(image_features, [1, 2],
                                                     keep_dims=True,
                                                     name='AvgPool')
    flattened_image_features = slim.flatten(spatial_averaged_image_features)
    if self._use_dropout:
      flattened_image_features = slim.dropout(flattened_image_features,
                                              keep_prob=self._dropout_keep_prob,
                                              is_training=self._is_training)
    with slim.arg_scope(self._fc_hyperparams):
      box_encodings = slim.fully_connected(
          flattened_image_features,
          self._num_classes * self._box_code_size,
          activation_fn=None,
          scope='BoxEncodingPredictor')
      class_predictions_with_background = slim.fully_connected(
          flattened_image_features,
          self._num_classes + 1,
          activation_fn=None,
          scope='ClassPredictor')
    box_encodings = tf.reshape(
        box_encodings, [-1, 1, self._num_classes, self._box_code_size])
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background, [-1, 1, self._num_classes + 1])

    predictions_dict = {
        BOX_ENCODINGS: box_encodings,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_with_background
    }

    if self._predict_instance_masks:
      with slim.arg_scope(self._conv_hyperparams):
        upsampled_features = slim.conv2d_transpose(
            image_features,
            num_outputs=self._mask_prediction_conv_depth,
            kernel_size=[2, 2],
            stride=2)
        mask_predictions = slim.conv2d(upsampled_features,
                                       num_outputs=self.num_classes,
                                       activation_fn=None,
                                       kernel_size=[1, 1])
        instance_masks = tf.expand_dims(tf.transpose(mask_predictions,
                                                     perm=[0, 3, 1, 2]),
                                        axis=1,
                                        name='MaskPredictor')
      predictions_dict[MASK_PREDICTIONS] = instance_masks
    return predictions_dict


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
               conv_hyperparams,
               min_depth,
               max_depth,
               num_layers_before_predictor,
               use_dropout,
               dropout_keep_prob,
               kernel_size,
               box_code_size,
               apply_sigmoid_to_scores=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams: Slim arg_scope with hyperparameters for convolution ops.
      min_depth: Minumum feature depth prior to predicting box encodings
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

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
    if min_depth > max_depth:
      raise ValueError('min_depth should be less than or equal to max_depth')
    self._conv_hyperparams = conv_hyperparams
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._use_dropout = use_dropout
    self._kernel_size = kernel_size
    self._box_code_size = box_code_size
    self._dropout_keep_prob = dropout_keep_prob
    self._apply_sigmoid_to_scores = apply_sigmoid_to_scores

  def _predict(self, image_features, num_predictions_per_location):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.

    Returns:
      A dictionary containing the following tensors.
        box_encodings: A float tensor of shape [batch_size, num_anchors, 1,
          code_size] representing the location of the objects, where
          num_anchors = feat_height * feat_width * num_predictions_per_location
        class_predictions_with_background: A float tensor of shape
          [batch_size, num_anchors, num_classes + 1] representing the class
          predictions for the proposals.
    """
    features_depth = static_shape.get_depth(image_features.get_shape())
    depth = max(min(features_depth, self._max_depth), self._min_depth)

    # Add a slot for the background class.
    num_class_slots = self.num_classes + 1
    net = image_features
    with slim.arg_scope(self._conv_hyperparams), \
         slim.arg_scope([slim.dropout], is_training=self._is_training):
      # Add additional conv layers before the predictor.
      if depth > 0 and self._num_layers_before_predictor > 0:
        for i in range(self._num_layers_before_predictor):
          net = slim.conv2d(
              net, depth, [1, 1], scope='Conv2d_%d_1x1_%d' % (i, depth))
      with slim.arg_scope([slim.conv2d], activation_fn=None,
                          normalizer_fn=None, normalizer_params=None):
        box_encodings = slim.conv2d(
            net, num_predictions_per_location * self._box_code_size,
            [self._kernel_size, self._kernel_size],
            scope='BoxEncodingPredictor')
        if self._use_dropout:
          net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
        class_predictions_with_background = slim.conv2d(
            net, num_predictions_per_location * num_class_slots,
            [self._kernel_size, self._kernel_size], scope='ClassPredictor')
        if self._apply_sigmoid_to_scores:
          class_predictions_with_background = tf.sigmoid(
              class_predictions_with_background)

    batch_size = static_shape.get_batch_size(image_features.get_shape())
    if batch_size is None:
      features_height = static_shape.get_height(image_features.get_shape())
      features_width = static_shape.get_width(image_features.get_shape())
      flattened_predictions_size = (features_height * features_width *
                                    num_predictions_per_location)
      box_encodings = tf.reshape(
          box_encodings,
          [-1, flattened_predictions_size, 1, self._box_code_size])
      class_predictions_with_background = tf.reshape(
          class_predictions_with_background,
          [-1, flattened_predictions_size, num_class_slots])
    else:
      box_encodings = tf.reshape(
          box_encodings, [batch_size, -1, 1, self._box_code_size])
      class_predictions_with_background = tf.reshape(
          class_predictions_with_background, [batch_size, -1, num_class_slots])
    return {BOX_ENCODINGS: box_encodings,
            CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background}
