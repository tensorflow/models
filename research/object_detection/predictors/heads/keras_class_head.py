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
import tensorflow as tf

from object_detection.predictors.heads import head


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
    """
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
    # Add a slot for the background class.
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
