# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Class Head.

Contains Class prediction head classes for different meta architectures.
All the class prediction heads have a predict function that receives the
`features` as the first argument and returns class predictions with background.
"""
import tensorflow.compat.v1 as tf

from object_detection.predictors.heads import head
from object_detection.utils import shape_utils


class ConvolutionalClassHead(head.KerasHead):
  """Convolutional class prediction head."""

  def __init__(self,
               is_training,
               num_class_slots,
               use_dropout,
               dropout_keep_prob,
               kernel_size,
               num_predictions_per_location,
               conv_hyperparams,
               freeze_batchnorm,
               class_prediction_bias_init=0.0,
               use_depthwise=False,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_class_slots: number of class slots. Note that num_class_slots may or
        may not include an implicit background category.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      num_predictions_per_location: Number of box predictions to be made per
        spatial location. Int specifying number of boxes per location.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.

    Raises:
      ValueError: if min_depth > max_depth.
      ValueError: if use_depthwise is True and kernel_size is 1.
    """
    if use_depthwise and (kernel_size == 1):
      raise ValueError('Should not use 1x1 kernel when using depthwise conv')

    super(ConvolutionalClassHead, self).__init__(name=name)
    self._is_training = is_training
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._kernel_size = kernel_size
    self._class_prediction_bias_init = class_prediction_bias_init
    self._use_depthwise = use_depthwise
    self._num_class_slots = num_class_slots

    self._class_predictor_layers = []

    if self._use_dropout:
      self._class_predictor_layers.append(
          # The Dropout layer's `training` parameter for the call method must
          # be set implicitly by the Keras set_learning_phase. The object
          # detection training code takes care of this.
          tf.keras.layers.Dropout(rate=1.0 - self._dropout_keep_prob))
    if self._use_depthwise:
      self._class_predictor_layers.append(
          tf.keras.layers.DepthwiseConv2D(
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              depth_multiplier=1,
              strides=1,
              dilation_rate=1,
              name='ClassPredictor_depthwise',
              **conv_hyperparams.params()))
      self._class_predictor_layers.append(
          conv_hyperparams.build_batch_norm(
              training=(is_training and not freeze_batchnorm),
              name='ClassPredictor_depthwise_batchnorm'))
      self._class_predictor_layers.append(
          conv_hyperparams.build_activation_layer(
              name='ClassPredictor_depthwise_activation'))
      self._class_predictor_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._num_class_slots, [1, 1],
              name='ClassPredictor',
              **conv_hyperparams.params(use_bias=True)))
    else:
      self._class_predictor_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._num_class_slots,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='ClassPredictor',
              bias_initializer=tf.constant_initializer(
                  self._class_prediction_bias_init),
              **conv_hyperparams.params(use_bias=True)))

  def _predict(self, features):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.

    Returns:
      class_predictions_with_background: A float tensor of shape
        [batch_size, num_anchors, num_class_slots] representing the class
        predictions for the proposals.
    """
    class_predictions_with_background = features
    for layer in self._class_predictor_layers:
      class_predictions_with_background = layer(
          class_predictions_with_background)
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background,
        [batch_size, -1, self._num_class_slots])
    return class_predictions_with_background


class MaskRCNNClassHead(head.KerasHead):
  """Mask RCNN class prediction head.

  This is a piece of Mask RCNN which is responsible for predicting
  just the class scores of boxes.

  Please refer to Mask RCNN paper:
  https://arxiv.org/abs/1703.06870
  """

  def __init__(self,
               is_training,
               num_class_slots,
               fc_hyperparams,
               freeze_batchnorm,
               use_dropout,
               dropout_keep_prob,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_class_slots: number of class slots. Note that num_class_slots may or
        may not include an implicit background category.
      fc_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for fully connected dense ops.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      use_dropout: Option to use dropout or not.  Note that a single dropout
        op is applied here prior to both box and class predictions, which stands
        in contrast to the ConvolutionalBoxPredictor below.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      name: A string name scope to assign to the class head. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(MaskRCNNClassHead, self).__init__(name=name)
    self._is_training = is_training
    self._freeze_batchnorm = freeze_batchnorm
    self._num_class_slots = num_class_slots
    self._fc_hyperparams = fc_hyperparams
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob

    self._class_predictor_layers = [tf.keras.layers.Flatten()]

    if self._use_dropout:
      self._class_predictor_layers.append(
          tf.keras.layers.Dropout(rate=1.0 - self._dropout_keep_prob))

    self._class_predictor_layers.append(
        tf.keras.layers.Dense(self._num_class_slots,
                              name='ClassPredictor_dense'))
    self._class_predictor_layers.append(
        fc_hyperparams.build_batch_norm(training=(is_training and
                                                  not freeze_batchnorm),
                                        name='ClassPredictor_batchnorm'))

  def _predict(self, features):
    """Predicts the class scores for boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing features for a batch of images.

    Returns:
      class_predictions_with_background: A float tensor of shape
        [batch_size, 1, num_class_slots] representing the class predictions for
        the proposals.
    """
    spatial_averaged_roi_pooled_features = tf.reduce_mean(
        features, [1, 2], keep_dims=True, name='AvgPool')
    net = spatial_averaged_roi_pooled_features
    for layer in self._class_predictor_layers:
      net = layer(net)
    class_predictions_with_background = tf.reshape(
        net,
        [-1, 1, self._num_class_slots])
    return class_predictions_with_background


class WeightSharedConvolutionalClassHead(head.KerasHead):
  """Weight shared convolutional class prediction head.

  This head allows sharing the same set of parameters (weights) when called more
  then once on different feature maps.
  """

  def __init__(self,
               num_class_slots,
               num_predictions_per_location,
               conv_hyperparams,
               kernel_size=3,
               class_prediction_bias_init=0.0,
               use_dropout=False,
               dropout_keep_prob=0.8,
               use_depthwise=False,
               apply_conv_hyperparams_to_heads=False,
               score_converter_fn=tf.identity,
               return_flat_predictions=True,
               name=None):
    """Constructor.

    Args:
      num_class_slots: number of class slots. Note that num_class_slots may or
        may not include an implicit background category.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location. Int specifying number of boxes per location.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      kernel_size: Size of final convolution kernel.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.
      use_dropout: Whether to apply dropout to class prediction head.
      dropout_keep_prob: Probability of keeping activiations.
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.
      apply_conv_hyperparams_to_heads: Whether to apply conv_hyperparams to
        depthwise seperable convolution layers in the box and class heads. By
        default, the conv_hyperparams are only applied to layers in the
        predictor tower when using depthwise separable convolutions.
      score_converter_fn: Callable elementwise nonlinearity (that takes tensors
        as inputs and returns tensors).
      return_flat_predictions: If true, returns flattened prediction tensor
        of shape [batch, height * width * num_predictions_per_location,
        box_coder]. Otherwise returns the prediction tensor before reshaping,
        whose shape is [batch, height, width, num_predictions_per_location *
        num_class_slots].
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.

    Raises:
      ValueError: if use_depthwise is True and kernel_size is 1.
    """
    if use_depthwise and (kernel_size == 1):
      raise ValueError('Should not use 1x1 kernel when using depthwise conv')

    super(WeightSharedConvolutionalClassHead, self).__init__(name=name)
    self._num_class_slots = num_class_slots
    self._num_predictions_per_location = num_predictions_per_location
    self._kernel_size = kernel_size
    self._class_prediction_bias_init = class_prediction_bias_init
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._use_depthwise = use_depthwise
    self._apply_conv_hyperparams_to_heads = apply_conv_hyperparams_to_heads
    self._score_converter_fn = score_converter_fn
    self._return_flat_predictions = return_flat_predictions

    self._class_predictor_layers = []

    if self._use_dropout:
      self._class_predictor_layers.append(
          tf.keras.layers.Dropout(rate=1.0 - self._dropout_keep_prob))
    if self._use_depthwise:
      kwargs = conv_hyperparams.params(use_bias=True)
      if self._apply_conv_hyperparams_to_heads:
        kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
        kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
        kwargs['pointwise_regularizer'] = kwargs['kernel_regularizer']
        kwargs['pointwise_initializer'] = kwargs['kernel_initializer']
      self._class_predictor_layers.append(
          tf.keras.layers.SeparableConv2D(
              num_predictions_per_location * self._num_class_slots,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              depth_multiplier=1,
              strides=1,
              name='ClassPredictor',
              bias_initializer=tf.constant_initializer(
                  self._class_prediction_bias_init),
              **kwargs))
    else:
      self._class_predictor_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._num_class_slots,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='ClassPredictor',
              bias_initializer=tf.constant_initializer(
                  self._class_prediction_bias_init),
              **conv_hyperparams.params(use_bias=True)))

  def _predict(self, features):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.

    Returns:
      class_predictions_with_background: A float tensor of shape
        [batch_size, num_anchors, num_class_slots] representing the class
        predictions for the proposals.
    """
    class_predictions_with_background = features
    for layer in self._class_predictor_layers:
      class_predictions_with_background = layer(
          class_predictions_with_background)
    batch_size, height, width = shape_utils.combined_static_and_dynamic_shape(
        features)[0:3]
    class_predictions_with_background = tf.reshape(
        class_predictions_with_background, [
            batch_size, height, width, self._num_predictions_per_location,
            self._num_class_slots
        ])
    class_predictions_with_background = self._score_converter_fn(
        class_predictions_with_background)
    if self._return_flat_predictions:
      class_predictions_with_background = tf.reshape(
          class_predictions_with_background,
          [batch_size, -1, self._num_class_slots])
    else:
      class_predictions_with_background = tf.reshape(
          class_predictions_with_background, [
              batch_size, height, width,
              self._num_predictions_per_location * self._num_class_slots
          ])
    return class_predictions_with_background
