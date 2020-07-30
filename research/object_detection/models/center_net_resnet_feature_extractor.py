# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Resnetv2 based feature extractors for CenterNet[1] meta architecture.

[1]: https://arxiv.org/abs/1904.07850
"""


import tensorflow.compat.v1 as tf

from object_detection.meta_architectures.center_net_meta_arch import CenterNetFeatureExtractor


class CenterNetResnetFeatureExtractor(CenterNetFeatureExtractor):
  """Resnet v2 base feature extractor for the CenterNet model."""

  def __init__(self, resnet_type, channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.), bgr_ordering=False):
    """Initializes the feature extractor with a specific ResNet architecture.

    Args:
      resnet_type: A string specifying which kind of ResNet to use. Currently
        only `resnet_v2_50` and `resnet_v2_101` are supported.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.

    """

    super(CenterNetResnetFeatureExtractor, self).__init__(
        channel_means=channel_means, channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
    if resnet_type == 'resnet_v2_101':
      self._base_model = tf.keras.applications.ResNet101V2(weights=None)
      output_layer = 'conv5_block3_out'
    elif resnet_type == 'resnet_v2_50':
      self._base_model = tf.keras.applications.ResNet50V2(weights=None)
      output_layer = 'conv5_block3_out'
    else:
      raise ValueError('Unknown Resnet Model {}'.format(resnet_type))
    output_layer = self._base_model.get_layer(output_layer)

    self._resnet_model = tf.keras.models.Model(inputs=self._base_model.input,
                                               outputs=output_layer.output)
    resnet_output = self._resnet_model(self._base_model.input)

    for num_filters in [256, 128, 64]:
      # TODO(vighneshb) This section has a few differences from the paper
      # Figure out how much of a performance impact they have.

      # 1. We use a simple convolution instead of a deformable convolution
      conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3,
                                    strides=1, padding='same')
      resnet_output = conv(resnet_output)
      resnet_output = tf.keras.layers.BatchNormalization()(resnet_output)
      resnet_output = tf.keras.layers.ReLU()(resnet_output)

      # 2. We use the default initialization for the convolution layers
      # instead of initializing it to do bilinear upsampling.
      conv_transpose = tf.keras.layers.Conv2DTranspose(filters=num_filters,
                                                       kernel_size=3, strides=2,
                                                       padding='same')
      resnet_output = conv_transpose(resnet_output)
      resnet_output = tf.keras.layers.BatchNormalization()(resnet_output)
      resnet_output = tf.keras.layers.ReLU()(resnet_output)

    self._feature_extractor_model = tf.keras.models.Model(
        inputs=self._base_model.input, outputs=resnet_output)

  def preprocess(self, resized_inputs):
    """Preprocess input images for the ResNet model.

    This scales images in the range [0, 255] to the range [-1, 1]

    Args:
      resized_inputs: a [batch, height, width, channels] float32 tensor.

    Returns:
      outputs: a [batch, height, width, channels] float32 tensor.

    """
    resized_inputs = super(CenterNetResnetFeatureExtractor, self).preprocess(
        resized_inputs)
    return tf.keras.applications.resnet_v2.preprocess_input(resized_inputs)

  def load_feature_extractor_weights(self, path):
    self._base_model.load_weights(path)

  def call(self, inputs):
    """Returns image features extracted by the backbone.

    Args:
      inputs: An image tensor of shape [batch_size, input_height,
        input_width, 3]

    Returns:
      features_list: A list of length 1 containing a tensor of shape
        [batch_size, input_height // 4, input_width // 4, 64] containing
        the features extracted by the ResNet.
    """
    return [self._feature_extractor_model(inputs)]

  @property
  def num_feature_outputs(self):
    return 1

  @property
  def out_stride(self):
    return 4

  def get_sub_model(self, sub_model_type):
    if sub_model_type == 'classification':
      return self._base_model
    else:
      supported_types = ['classification']
      raise ValueError(
          ('Sub model {} is not defined for ResNet.'.format(sub_model_type)
           + 'Supported types are {}.'.format(supported_types)
           + 'Use the script convert_keras_models.py to create your own '
           + 'classification checkpoints.'))


def resnet_v2_101(channel_means, channel_stds, bgr_ordering):
  """The ResNet v2 101 feature extractor."""

  return CenterNetResnetFeatureExtractor(
      resnet_type='resnet_v2_101',
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering
  )


def resnet_v2_50(channel_means, channel_stds, bgr_ordering):
  """The ResNet v2 50 feature extractor."""

  return CenterNetResnetFeatureExtractor(
      resnet_type='resnet_v2_50',
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)
