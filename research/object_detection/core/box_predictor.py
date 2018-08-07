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
from object_detection.utils import shape_utils
from object_detection.utils import static_shape
#import tensorflow.contrib.ndlstm as ndlstm
import object_detection.core.lstm2d as lstm2d

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
  def is_keras_model(self):
    return False

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


class KerasBoxPredictor(tf.keras.Model):
  """Keras-based BoxPredictor."""

  def __init__(self, is_training, num_classes, freeze_batchnorm,
               inplace_batchnorm_update, name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(KerasBoxPredictor, self).__init__(name=name)

    self._is_training = is_training
    self._num_classes = num_classes
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update

  @property
  def is_keras_model(self):
    return True

  @property
  def num_classes(self):
    return self._num_classes

  def call(self, image_features, **kwargs):
    """Computes encoded object locations and corresponding confidences.

    Takes a list of high level image feature maps as input and produces a list
    of box encodings and a list of class scores where each element in the output
    lists correspond to the feature maps in the input list.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
      width_i, channels_i] containing features for a batch of images.
      **kwargs: Additional keyword arguments for specific implementations of
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
    return self._predict(image_features, **kwargs)

  @abstractmethod
  def _predict(self, image_features, **kwargs):
    """Implementations must override this method.

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
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.

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
    self._class_prediction_bias_init = class_prediction_bias_init


  def _predict(self, image_features, num_predictions_per_location):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      num_predictions_per_location: an integer representing the number of box
        predictions to be made per spatial location in the feature map.

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
    # Add a slot for the background class.
    num_class_slots = self.num_classes + 1
    net = image_features
    with slim.arg_scope(self._conv_hyperparams), \
         slim.arg_scope([slim.dropout], is_training=self._is_training):
      # Add additional conv layers before the class predictor.
      features_depth = static_shape.get_depth(image_features.get_shape())
      depth = max(min(features_depth, self._max_depth), self._min_depth)
      tf.logging.info('depth of additional conv before box predictor: {}'.
                      format(depth))
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
            [self._kernel_size, self._kernel_size], scope='ClassPredictor',
            biases_initializer=tf.constant_initializer(
                self._class_prediction_bias_init))
        if self._apply_sigmoid_to_scores:
          class_predictions_with_background = tf.sigmoid(
              class_predictions_with_background)

    combined_feature_map_shape = shape_utils.combined_static_and_dynamic_shape(
        image_features)
    box_encodings = tf.reshape(
        box_encodings, tf.stack([combined_feature_map_shape[0],
                                 combined_feature_map_shape[1] *
                                 combined_feature_map_shape[2] *
                                 num_predictions_per_location,
                                 1, self._box_code_size]))
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background,
        tf.stack([combined_feature_map_shape[0],
                  combined_feature_map_shape[1] *
                  combined_feature_map_shape[2] *
                  num_predictions_per_location,
                  num_class_slots]))
    return {BOX_ENCODINGS: box_encodings,
            CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background}


  #def _predict(self, image_features, num_predictions_per_location):
  #  """Computes encoded object locations and corresponding confidences.

  #  Args:
  #    image_features: A float tensor of shape [batch_size, height, width,
  #      channels] containing features for a batch of images.
  #    num_predictions_per_location: an integer representing the number of box
  #      predictions to be made per spatial location in the feature map.

  #  Returns:
  #    A dictionary containing the following tensors.
  #      box_encodings: A float tensor of shape [batch_size, num_anchors, 1,
  #        code_size] representing the location of the objects, where
  #        num_anchors = feat_height * feat_width * num_predictions_per_location
  #      class_predictions_with_background: A float tensor of shape
  #        [batch_size, num_anchors, num_classes + 1] representing the class
  #        predictions for the proposals.
  #  """
  #  # Add a slot for the background class.
  #  num_class_slots = self.num_classes + 1
  #  net = image_features
  #  with slim.arg_scope(self._conv_hyperparams), \
  #       slim.arg_scope([slim.dropout], is_training=self._is_training):
  #    # Add additional conv layers before the class predictor.
  #    features_depth = static_shape.get_depth(image_features.get_shape())
  #    depth = max(min(features_depth, self._max_depth), self._min_depth)
  #    tf.logging.info('depth of additional conv before box predictor: {}'.
  #                    format(depth))
  #    if depth > 0 and self._num_layers_before_predictor > 0:
  #      for i in range(self._num_layers_before_predictor):
  #        net = slim.conv2d(
  #            net, depth, [1, 1], scope='Conv2d_%d_1x1_%d' % (i, depth))

  #    #box_encodings_class_predictions =  lstm2d.separable_lstm(net, (num_predictions_per_location * self._box_code_size) + (num_predictions_per_location * num_class_slots), kernel_size=None, nhidden=256, scope='lstm2d-1')      
  #    #w_var = tf.Variable(initial_value=0.1,name="context_scale_var")
  #    #context_net = w_var *  lstm2d.separable_lstm(net, 256, kernel_size=None, nhidden=256, scope='lstm2d-1')      
  #    #net = tf.concat([net,context_net], axis=3)

  #    with slim.arg_scope([slim.conv2d], activation_fn=None,
  #                        normalizer_fn=None, normalizer_params=None):

  #      #box_encodings = box_encodings_class_predictions[:,:,:,0:num_predictions_per_location * self._box_code_size]
  #      #box_encodings = tf.identity(box_encodings,"BoxEncodingPredictor")
  #      box_encodings = slim.conv2d(
  #          net, num_predictions_per_location * self._box_code_size,
  #          [self._kernel_size, self._kernel_size],
  #          scope='BoxEncodingPredictor')
  #      if self._use_dropout:
  #        net = slim.dropout(net, keep_prob=self._dropout_keep_prob)

  #      #class_predictions_with_background = box_encodings_class_predictions[:,:,:,-(num_predictions_per_location * num_class_slots):]
  #      #class_predictions_with_background = tf.identity(class_predictions_with_background,"ClassPredictor")

  #      class_predictions_with_background = slim.conv2d(
  #          net, num_predictions_per_location * num_class_slots,
  #          [self._kernel_size, self._kernel_size], scope='ClassPredictor',
  #          biases_initializer=tf.constant_initializer(
  #              self._class_prediction_bias_init))
  #      if self._apply_sigmoid_to_scores:
  #        class_predictions_with_background = tf.sigmoid(
  #            class_predictions_with_background)

  #  combined_feature_map_shape = shape_utils.combined_static_and_dynamic_shape(
  #      image_features)
  #  box_encodings = tf.reshape(
  #      box_encodings, tf.stack([combined_feature_map_shape[0],
  #                               combined_feature_map_shape[1] *
  #                               combined_feature_map_shape[2] *
  #                               num_predictions_per_location,
  #                               1, self._box_code_size]))    
  #  class_predictions_with_background = tf.reshape(
  #      class_predictions_with_background,
  #      tf.stack([combined_feature_map_shape[0],
  #                combined_feature_map_shape[1] *
  #                combined_feature_map_shape[2] *
  #                num_predictions_per_location,
  #                num_class_slots]))
  #  return {BOX_ENCODINGS: box_encodings,
  #          CLASS_PREDICTIONS_WITH_BACKGROUND:
  #          class_predictions_with_background}


class FpnSharedWeightsConvolutionalBoxPredictor(BoxPredictor):
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
               apply_sigmoid_to_scores=False,
               class_prediction_bias_init=0.0):
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
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(FpnSharedWeightsConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
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
    self._class_prediction_bias_init = class_prediction_bias_init

  def _predict(self, fpn_image_features, num_predictions_per_location):
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
    # Add a slot for the background class.
    num_class_slots = self.num_classes + 1
    box_encodings_list = []
    class_predictions_with_background_list = []
    with tf.variable_scope('WeightSharedConvolutionalBoxPredictor', reuse=tf.AUTO_REUSE):
      with slim.arg_scope(self._conv_hyperparams), \
           slim.arg_scope([slim.dropout], is_training=self._is_training):
        for feature_map_index in range(len(fpn_image_features)):
          feature_map = fpn_image_features[feature_map_index]
          net = feature_map  
          # Add additional conv layers before the class predictor.
          features_depth = static_shape.get_depth(feature_map.get_shape())
          depth = max(min(features_depth, self._max_depth), self._min_depth)
          tf.logging.info('depth of additional conv before box predictor: {}'.
                        format(depth))
          if depth > 0 and self._num_layers_before_predictor > 0:
            for i in range(self._num_layers_before_predictor):
              net = slim.conv2d(net, depth, [1, 1], scope='Conv2d_%d_1x1_%d' % (i, depth))

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
                [self._kernel_size, self._kernel_size], scope='ClassPredictor',
                biases_initializer=tf.constant_initializer(
                    self._class_prediction_bias_init))
            if self._apply_sigmoid_to_scores:
               class_predictions_with_background = tf.sigmoid(
                 class_predictions_with_background)

          combined_feature_map_shape = shape_utils.combined_static_and_dynamic_shape(feature_map)
          box_encodings = tf.reshape(box_encodings, tf.stack([combined_feature_map_shape[0], combined_feature_map_shape[1] * combined_feature_map_shape[2] *
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
          class_predictions_with_background_list.append(class_predictions_with_background)
    box_encodings = tf.concat(box_encodings_list,axis=1)
    class_predictions_with_background = tf.concat(class_predictions_with_background_list,axis=1)
    return {BOX_ENCODINGS: box_encodings,
            CLASS_PREDICTIONS_WITH_BACKGROUND:
            class_predictions_with_background}
