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
"""Hourglass[1] feature extractor for CenterNet[2] meta architecture.

[1]: https://arxiv.org/abs/1603.06937
[2]: https://arxiv.org/abs/1904.07850
"""

from object_detection.meta_architectures import center_net_meta_arch
from object_detection.models.keras_models import hourglass_network


class CenterNetHourglassFeatureExtractor(
    center_net_meta_arch.CenterNetFeatureExtractor):
  """The hourglass feature extractor for CenterNet.

  This class is a thin wrapper around the HourglassFeatureExtractor class
  along with some preprocessing methods inherited from the base class.
  """

  def __init__(self, hourglass_net, channel_means=(0., 0., 0.),
               channel_stds=(1., 1., 1.), bgr_ordering=False):
    """Intializes the feature extractor.

    Args:
      hourglass_net: The underlying hourglass network to use.
      channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it.
      channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
      bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
    """

    super(CenterNetHourglassFeatureExtractor, self).__init__(
        channel_means=channel_means, channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)
    self._network = hourglass_net

  def call(self, inputs):
    return self._network(inputs)

  @property
  def out_stride(self):
    """The stride in the output image of the network."""
    return 4

  @property
  def num_feature_outputs(self):
    """Ther number of feature outputs returned by the feature extractor."""
    return self._network.num_hourglasses

  @property
  def supported_sub_model_types(self):
    return ['detection']

  def get_sub_model(self, sub_model_type):
    if sub_model_type == 'detection':
      return self._network
    else:
      ValueError('Sub model type "{}" not supported.'.format(sub_model_type))


def hourglass_10(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The Hourglass-10 backbone for CenterNet."""
  del kwargs

  network = hourglass_network.hourglass_10(num_channels=32)
  return CenterNetHourglassFeatureExtractor(
      network, channel_means=channel_means, channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)


def hourglass_20(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The Hourglass-20 backbone for CenterNet."""
  del kwargs

  network = hourglass_network.hourglass_20(num_channels=48)
  return CenterNetHourglassFeatureExtractor(
      network, channel_means=channel_means, channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)


def hourglass_32(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The Hourglass-32 backbone for CenterNet."""
  del kwargs

  network = hourglass_network.hourglass_32(num_channels=48)
  return CenterNetHourglassFeatureExtractor(
      network, channel_means=channel_means, channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)


def hourglass_52(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The Hourglass-52 backbone for CenterNet."""
  del kwargs

  network = hourglass_network.hourglass_52(num_channels=64)
  return CenterNetHourglassFeatureExtractor(
      network, channel_means=channel_means, channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)


def hourglass_104(channel_means, channel_stds, bgr_ordering, **kwargs):
  """The Hourglass-104 backbone for CenterNet."""
  del kwargs

  # TODO(vighneshb): update hourglass_104 signature to match with other
  # hourglass networks.
  network = hourglass_network.hourglass_104()
  return CenterNetHourglassFeatureExtractor(
      network, channel_means=channel_means, channel_stds=channel_stds,
      bgr_ordering=bgr_ordering)
