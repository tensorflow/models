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

"""Box Head.

Contains Box prediction head classes for different meta architectures.
All the box prediction heads have a _predict function that receives the
`features` as the first argument and returns `box_encodings`.
"""
import tensorflow as tf

from object_detection.predictors.heads import head


class ConvolutionalBoxHead(head.KerasHead):
  """Convolutional box prediction head."""

  def __init__(self,
               is_training,
               box_code_size,
               kernel_size,
               num_predictions_per_location,
               conv_hyperparams,
               freeze_batchnorm,
               use_depthwise=True,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      box_code_size: Size of encoding for each box.
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
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxHead, self).__init__(name=name)
    self._is_training = is_training
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._num_predictions_per_location = num_predictions_per_location
    self._use_depthwise = use_depthwise

    self._box_encoder_layers = []

    if self._use_depthwise:
      self._box_encoder_layers.append(
          tf.keras.layers.DepthwiseConv2D(
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              depth_multiplier=1,
              strides=1,
              dilation_rate=1,
              name='BoxEncodingPredictor_depthwise',
              **conv_hyperparams.params()))
      self._box_encoder_layers.append(
          conv_hyperparams.build_batch_norm(
              training=(is_training and not freeze_batchnorm),
              name='BoxEncodingPredictor_depthwise_batchnorm'))
      self._box_encoder_layers.append(
          conv_hyperparams.build_activation_layer(
              name='BoxEncodingPredictor_depthwise_activation'))
      self._box_encoder_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._box_code_size, [1, 1],
              name='BoxEncodingPredictor',
              **conv_hyperparams.params(activation=None)))
    else:
      self._box_encoder_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._box_code_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='BoxEncodingPredictor',
              **conv_hyperparams.params(activation=None)))

  def _predict(self, features):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, num_anchors, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes.
    """
    box_encodings = features
    for layer in self._box_encoder_layers:
      box_encodings = layer(box_encodings)
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    box_encodings = tf.reshape(box_encodings,
                               [batch_size, -1, 1, self._box_code_size])
    return box_encodings
