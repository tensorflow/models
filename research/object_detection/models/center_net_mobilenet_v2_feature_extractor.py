# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""MobileNet V2[1] feature extractor for CenterNet[2] meta architecture.

[1]: https://arxiv.org/abs/1801.04381
[2]: https://arxiv.org/abs/1904.07850
"""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import center_net_meta_arch
from object_detection.models.keras_models import mobilenet_v2 as mobilenetv2


class CenterNetMobileNetV2FeatureExtractor(
    center_net_meta_arch.CenterNetFeatureExtractor):
  """The MobileNet V2 feature extractor for CenterNet."""

  def __init__(self,
               mobilenet_v2_net,
               channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.),
               bgr_ordering=False):
    """Intializes the feature extractor.

    Args:
      mobilenet_v2_net: The underlying mobilenet_v2 network to use.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
    """

    super(CenterNetMobileNetV2FeatureExtractor, self).__init__(
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
    self._network = mobilenet_v2_net

    output = self._network(self._network.input)

    # MobileNet by itself transforms a 224x224x3 volume into a 7x7x1280, which
    # leads to a stride of 32. We perform upsampling to get it to a target
    # stride of 4.
    for num_filters in [256, 128, 64]:
      # 1. We use a simple convolution instead of a deformable convolution
      conv = tf.keras.layers.Conv2D(
          filters=num_filters, kernel_size=1, strides=1, padding='same')
      output = conv(output)
      output = tf.keras.layers.BatchNormalization()(output)
      output = tf.keras.layers.ReLU()(output)

      # 2. We use the default initialization for the convolution layers
      # instead of initializing it to do bilinear upsampling.
      conv_transpose = tf.keras.layers.Conv2DTranspose(
          filters=num_filters, kernel_size=3, strides=2, padding='same')
      output = conv_transpose(output)
      output = tf.keras.layers.BatchNormalization()(output)
      output = tf.keras.layers.ReLU()(output)

    self._network = tf.keras.models.Model(
        inputs=self._network.input, outputs=output)

  def preprocess(self, resized_inputs):
    resized_inputs = super(CenterNetMobileNetV2FeatureExtractor,
                           self).preprocess(resized_inputs)
    return tf.keras.applications.mobilenet_v2.preprocess_input(resized_inputs)

  def load_feature_extractor_weights(self, path):
    self._network.load_weights(path)

  def get_base_model(self):
    return self._network

  def call(self, inputs):
    return [self._network(inputs)]

  @property
  def out_stride(self):
    """The stride in the output image of the network."""
    return 4

  @property
  def num_feature_outputs(self):
    """The number of feature outputs returned by the feature extractor."""
    return 1

  @property
  def supported_sub_model_types(self):
    return ['detection']

  def get_sub_model(self, sub_model_type):
    if sub_model_type == 'detection':
      return self._network
    else:
      ValueError('Sub model type "{}" not supported.'.format(sub_model_type))


def mobilenet_v2(channel_means, channel_stds, bgr_ordering,
                 depth_multiplier=1.0, **kwargs):
  """The MobileNetV2 backbone for CenterNet."""
  del kwargs

  # We set 'is_training' to True for now.
  network = mobilenetv2.mobilenet_v2(
      batchnorm_training=True,
      alpha=depth_multiplier,
      include_top=False,
      weights='imagenet' if depth_multiplier == 1.0 else None)
  return CenterNetMobileNetV2FeatureExtractor(
      network,
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)
