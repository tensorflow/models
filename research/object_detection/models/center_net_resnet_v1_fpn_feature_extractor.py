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
"""Resnetv1 FPN [1] based feature extractors for CenterNet[2] meta architecture.


[1]: https://arxiv.org/abs/1612.03144.
[2]: https://arxiv.org/abs/1904.07850.
"""
import tensorflow.compat.v1 as tf

from object_detection.meta_architectures.center_net_meta_arch import CenterNetFeatureExtractor
from object_detection.models.keras_models import resnet_v1


_RESNET_MODEL_OUTPUT_LAYERS = {
    'resnet_v1_18': ['conv2_block2_out', 'conv3_block2_out',
                     'conv4_block2_out', 'conv5_block2_out'],
    'resnet_v1_34': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_50': ['conv2_block3_out', 'conv3_block4_out',
                     'conv4_block6_out', 'conv5_block3_out'],
    'resnet_v1_101': ['conv2_block3_out', 'conv3_block4_out',
                      'conv4_block23_out', 'conv5_block3_out'],
}


class CenterNetResnetV1FpnFeatureExtractor(CenterNetFeatureExtractor):
  """Resnet v1 FPN base feature extractor for the CenterNet model.

  This feature extractor uses residual skip connections and nearest neighbor
  upsampling to produce an output feature map of stride 4, which has precise
  localization information along with strong semantic information from the top
  of the net. This design does not exactly follow the original FPN design,
  specifically:
  - Since only one output map is necessary for heatmap prediction (stride 4
    output), the top-down feature maps can have different numbers of channels.
    Specifically, the top down feature maps have the following sizes:
    [h/4, w/4, 64], [h/8, w/8, 128], [h/16, w/16, 256], [h/32, w/32, 256].
  - No additional coarse features are used after conv5_x.
  """

  def __init__(self, resnet_type, channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.), bgr_ordering=False):
    """Initializes the feature extractor with a specific ResNet architecture.

    Args:
      resnet_type: A string specifying which kind of ResNet to use. Currently
        only `resnet_v1_50` and `resnet_v1_101` are supported.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.

    """

    super(CenterNetResnetV1FpnFeatureExtractor, self).__init__(
        channel_means=channel_means, channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
    if resnet_type == 'resnet_v1_50':
      self._base_model = tf.keras.applications.ResNet50(weights=None,
                                                        include_top=False)
    elif resnet_type == 'resnet_v1_101':
      self._base_model = tf.keras.applications.ResNet101(weights=None,
                                                         include_top=False)
    elif resnet_type == 'resnet_v1_18':
      self._base_model = resnet_v1.resnet_v1_18(weights=None, include_top=False)
    elif resnet_type == 'resnet_v1_34':
      self._base_model = resnet_v1.resnet_v1_34(weights=None, include_top=False)
    else:
      raise ValueError('Unknown Resnet Model {}'.format(resnet_type))
    output_layers = _RESNET_MODEL_OUTPUT_LAYERS[resnet_type]
    outputs = [self._base_model.get_layer(output_layer_name).output
               for output_layer_name in output_layers]

    self._resnet_model = tf.keras.models.Model(inputs=self._base_model.input,
                                               outputs=outputs)
    resnet_outputs = self._resnet_model(self._base_model.input)

    # Construct the top-down feature maps.
    top_layer = resnet_outputs[-1]
    residual_op = tf.keras.layers.Conv2D(filters=256, kernel_size=1,
                                         strides=1, padding='same')
    top_down = residual_op(top_layer)

    num_filters_list = [256, 128, 64]
    for i, num_filters in enumerate(num_filters_list):
      level_ind = 2 - i
      # Upsample.
      upsample_op = tf.keras.layers.UpSampling2D(2, interpolation='nearest')
      top_down = upsample_op(top_down)

      # Residual (skip-connection) from bottom-up pathway.
      residual_op = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1,
                                           strides=1, padding='same')
      residual = residual_op(resnet_outputs[level_ind])

      # Merge.
      top_down = top_down + residual
      next_num_filters = num_filters_list[i+1] if i + 1 <= 2 else 64
      conv = tf.keras.layers.Conv2D(filters=next_num_filters,
                                    kernel_size=3, strides=1, padding='same')
      top_down = conv(top_down)
      top_down = tf.keras.layers.BatchNormalization()(top_down)
      top_down = tf.keras.layers.ReLU()(top_down)

    self._feature_extractor_model = tf.keras.models.Model(
        inputs=self._base_model.input, outputs=top_down)

  def preprocess(self, resized_inputs):
    """Preprocess input images for the ResNet model.

    This scales images in the range [0, 255] to the range [-1, 1]

    Args:
      resized_inputs: a [batch, height, width, channels] float32 tensor.

    Returns:
      outputs: a [batch, height, width, channels] float32 tensor.

    """
    resized_inputs = super(
        CenterNetResnetV1FpnFeatureExtractor, self).preprocess(resized_inputs)
    return tf.keras.applications.resnet.preprocess_input(resized_inputs)

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

  @property
  def classification_backbone(self):
    return self._base_model


def resnet_v1_101_fpn(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The ResNet v1 101 FPN feature extractor."""
  del kwargs

  return CenterNetResnetV1FpnFeatureExtractor(
      resnet_type='resnet_v1_101',
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering
  )


def resnet_v1_50_fpn(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The ResNet v1 50 FPN feature extractor."""
  del kwargs

  return CenterNetResnetV1FpnFeatureExtractor(
      resnet_type='resnet_v1_50',
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)


def resnet_v1_34_fpn(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The ResNet v1 34 FPN feature extractor."""
  del kwargs

  return CenterNetResnetV1FpnFeatureExtractor(
      resnet_type='resnet_v1_34',
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering
  )


def resnet_v1_18_fpn(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The ResNet v1 18 FPN feature extractor."""
  del kwargs

  return CenterNetResnetV1FpnFeatureExtractor(
      resnet_type='resnet_v1_18',
      channel_means=channel_means,
      channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)
