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
"""MobileNet V2[1] + FPN[2] feature extractor for CenterNet[3] meta architecture.

[1]: https://arxiv.org/abs/1801.04381
[2]: https://arxiv.org/abs/1612.03144.
[3]: https://arxiv.org/abs/1904.07850
"""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import center_net_meta_arch
from object_detection.models.keras_models import mobilenet_v2 as mobilenetv2


_MOBILENET_V2_FPN_SKIP_LAYERS = [
    'block_2_add', 'block_5_add', 'block_9_add', 'out_relu'
]


class CenterNetMobileNetV2FPNFeatureExtractor(
    center_net_meta_arch.CenterNetFeatureExtractor):
  """The MobileNet V2 with FPN skip layers feature extractor for CenterNet."""

  def __init__(self,
               mobilenet_v2_net,
               channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.),
               bgr_ordering=False,
               fpn_separable_conv=False):
    """Intializes the feature extractor.

    Args:
      mobilenet_v2_net: The underlying mobilenet_v2 network to use.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
      fpn_separable_conv: If set to True, all convolutional layers in the FPN
        network will be replaced by separable convolutions.
    """

    super(CenterNetMobileNetV2FPNFeatureExtractor, self).__init__(
        channel_means=channel_means,
        channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
    self._base_model = mobilenet_v2_net

    output = self._base_model(self._base_model.input)

    # Add pyramid feature network on every layer that has stride 2.
    skip_outputs = [
        self._base_model.get_layer(skip_layer_name).output
        for skip_layer_name in _MOBILENET_V2_FPN_SKIP_LAYERS
    ]
    self._fpn_model = tf.keras.models.Model(
        inputs=self._base_model.input, outputs=skip_outputs)
    fpn_outputs = self._fpn_model(self._base_model.input)

    # Construct the top-down feature maps -- we start with an output of
    # 7x7x1280, which we continually upsample, apply a residual on and merge.
    # This results in a 56x56x24 output volume.
    top_layer = fpn_outputs[-1]
    # Use normal convolutional layer since the kernel_size is 1.
    residual_op = tf.keras.layers.Conv2D(
        filters=64, kernel_size=1, strides=1, padding='same')
    top_down = residual_op(top_layer)

    num_filters_list = [64, 32, 24]
    for i, num_filters in enumerate(num_filters_list):
      level_ind = len(num_filters_list) - 1 - i
      # Upsample.
      upsample_op = tf.keras.layers.UpSampling2D(2, interpolation='nearest')
      top_down = upsample_op(top_down)

      # Residual (skip-connection) from bottom-up pathway.
      # Use normal convolutional layer since the kernel_size is 1.
      residual_op = tf.keras.layers.Conv2D(
          filters=num_filters, kernel_size=1, strides=1, padding='same')
      residual = residual_op(fpn_outputs[level_ind])

      # Merge.
      top_down = top_down + residual
      next_num_filters = num_filters_list[i + 1] if i + 1 <= 2 else 24
      if fpn_separable_conv:
        conv = tf.keras.layers.SeparableConv2D(
            filters=next_num_filters, kernel_size=3, strides=1, padding='same')
      else:
        conv = tf.keras.layers.Conv2D(
            filters=next_num_filters, kernel_size=3, strides=1, padding='same')
      top_down = conv(top_down)
      top_down = tf.keras.layers.BatchNormalization()(top_down)
      top_down = tf.keras.layers.ReLU()(top_down)

    output = top_down

    self._feature_extractor_model = tf.keras.models.Model(
        inputs=self._base_model.input, outputs=output)

  def preprocess(self, resized_inputs):
    resized_inputs = super(CenterNetMobileNetV2FPNFeatureExtractor,
                           self).preprocess(resized_inputs)
    return tf.keras.applications.mobilenet_v2.preprocess_input(resized_inputs)

  def load_feature_extractor_weights(self, path):
    self._base_model.load_weights(path)

  @property
  def supported_sub_model_types(self):
    return ['classification']

  def get_sub_model(self, sub_model_type):
    if sub_model_type == 'classification':
      return self._base_model
    else:
      ValueError('Sub model type "{}" not supported.'.format(sub_model_type))

  def call(self, inputs):
    return [self._feature_extractor_model(inputs)]

  @property
  def out_stride(self):
    """The stride in the output image of the network."""
    return 4

  @property
  def num_feature_outputs(self):
    """The number of feature outputs returned by the feature extractor."""
    return 1


def mobilenet_v2_fpn(channel_means, channel_stds, bgr_ordering):
  """The MobileNetV2+FPN backbone for CenterNet."""

  # Set to batchnorm_training to True for now.
  network = mobilenetv2.mobilenet_v2(batchnorm_training=True, include_top=False)
  return CenterNetMobileNetV2FPNFeatureExtractor(
      network,
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering,
      fpn_separable_conv=False)


def mobilenet_v2_fpn_sep_conv(channel_means, channel_stds, bgr_ordering):
  """Same as mobilenet_v2_fpn except with separable convolution in FPN."""

  # Setting batchnorm_training to True, which will use the correct
  # BatchNormalization layer strategy based on the current Keras learning phase.
  # TODO(yuhuic): expriment with True vs. False to understand it's effect in
  # practice.
  network = mobilenetv2.mobilenet_v2(batchnorm_training=True, include_top=False)
  return CenterNetMobileNetV2FPNFeatureExtractor(
      network,
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering,
      fpn_separable_conv=True)
