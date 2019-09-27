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
               use_depthwise=False,
               box_encodings_clip_range=None,
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
      box_encodings_clip_range: Min and max values for clipping box_encodings.
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
    self._box_encodings_clip_range = box_encodings_clip_range

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
              **conv_hyperparams.params(use_bias=True)))
    else:
      self._box_encoder_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._box_code_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='BoxEncodingPredictor',
              **conv_hyperparams.params(use_bias=True)))

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
    # Clipping the box encodings to make the inference graph TPU friendly.
    if self._box_encodings_clip_range is not None:
      box_encodings = tf.clip_by_value(
          box_encodings, self._box_encodings_clip_range.min,
          self._box_encodings_clip_range.max)
    box_encodings = tf.reshape(box_encodings,
                               [batch_size, -1, 1, self._box_code_size])
    return box_encodings


class MaskRCNNBoxHead(head.KerasHead):
  """Box prediction head.

  This is a piece of Mask RCNN which is responsible for predicting
  just the box encodings.

  Please refer to Mask RCNN paper:
  https://arxiv.org/abs/1703.06870
  """

  def __init__(self,
               is_training,
               num_classes,
               fc_hyperparams,
               freeze_batchnorm,
               use_dropout,
               dropout_keep_prob,
               box_code_size,
               share_box_across_classes=False,
               name=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
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
      box_code_size: Size of encoding for each box.
      share_box_across_classes: Whether to share boxes across classes rather
        than use a different box for each class.
      name: A string name scope to assign to the box head. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(MaskRCNNBoxHead, self).__init__(name=name)
    self._is_training = is_training
    self._num_classes = num_classes
    self._fc_hyperparams = fc_hyperparams
    self._freeze_batchnorm = freeze_batchnorm
    self._use_dropout = use_dropout
    self._dropout_keep_prob = dropout_keep_prob
    self._box_code_size = box_code_size
    self._share_box_across_classes = share_box_across_classes

    self._box_encoder_layers = [tf.keras.layers.Flatten()]

    if self._use_dropout:
      self._box_encoder_layers.append(
          tf.keras.layers.Dropout(rate=1.0 - self._dropout_keep_prob))

    self._number_of_boxes = 1
    if not self._share_box_across_classes:
      self._number_of_boxes = self._num_classes

    self._box_encoder_layers.append(
        tf.keras.layers.Dense(self._number_of_boxes * self._box_code_size,
                              name='BoxEncodingPredictor_dense'))
    self._box_encoder_layers.append(
        fc_hyperparams.build_batch_norm(training=(is_training and
                                                  not freeze_batchnorm),
                                        name='BoxEncodingPredictor_batchnorm'))

  def _predict(self, features):
    """Predicts box encodings.

    Args:
      features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, 1, num_classes, code_size] representing the location of the
        objects.
    """
    spatial_averaged_roi_pooled_features = tf.reduce_mean(
        features, [1, 2], keep_dims=True, name='AvgPool')
    net = spatial_averaged_roi_pooled_features
    for layer in self._box_encoder_layers:
      net = layer(net)
    box_encodings = tf.reshape(net,
                               [-1, 1,
                                self._number_of_boxes,
                                self._box_code_size])
    return box_encodings


# TODO(b/128922690): Unify the implementations of ConvolutionalBoxHead
# and WeightSharedConvolutionalBoxHead
class WeightSharedConvolutionalBoxHead(head.KerasHead):
  """Weight shared convolutional box prediction head based on Keras.

  This head allows sharing the same set of parameters (weights) when called more
  then once on different feature maps.
  """

  def __init__(self,
               box_code_size,
               num_predictions_per_location,
               conv_hyperparams,
               kernel_size=3,
               use_depthwise=False,
               box_encodings_clip_range=None,
               return_flat_predictions=True,
               name=None):
    """Constructor.

    Args:
      box_code_size: Size of encoding for each box.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location. Int specifying number of boxes per location.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      kernel_size: Size of final convolution kernel.
      use_depthwise: Whether to use depthwise convolutions for prediction steps.
        Default is False.
      box_encodings_clip_range: Min and max values for clipping box_encodings.
      return_flat_predictions: If true, returns flattened prediction tensor
        of shape [batch, height * width * num_predictions_per_location,
        box_coder]. Otherwise returns the prediction tensor before reshaping,
        whose shape is [batch, height, width, num_predictions_per_location *
        num_class_slots].
      name: A string name scope to assign to the model. If `None`, Keras
        will auto-generate one from the class name.
    """
    super(WeightSharedConvolutionalBoxHead, self).__init__(name=name)
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._num_predictions_per_location = num_predictions_per_location
    self._use_depthwise = use_depthwise
    self._box_encodings_clip_range = box_encodings_clip_range
    self._return_flat_predictions = return_flat_predictions

    self._box_encoder_layers = []

    if self._use_depthwise:
      self._box_encoder_layers.append(
          tf.keras.layers.SeparableConv2D(
              num_predictions_per_location * self._box_code_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='BoxPredictor',
              **conv_hyperparams.params(use_bias=True)))
    else:
      self._box_encoder_layers.append(
          tf.keras.layers.Conv2D(
              num_predictions_per_location * self._box_code_size,
              [self._kernel_size, self._kernel_size],
              padding='SAME',
              name='BoxPredictor',
              **conv_hyperparams.params(use_bias=True)))

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
    # Clipping the box encodings to make the inference graph TPU friendly.
    if self._box_encodings_clip_range is not None:
      box_encodings = tf.clip_by_value(
          box_encodings, self._box_encodings_clip_range.min,
          self._box_encodings_clip_range.max)
    if self._return_flat_predictions:
      box_encodings = tf.reshape(box_encodings,
                                 [batch_size, -1, self._box_code_size])
    return box_encodings
