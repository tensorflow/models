# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of box sampler."""

# Import libraries
import tensorflow as tf

from official.vision.beta.ops import sampling_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class BoxSampler(tf.keras.layers.Layer):
  """Creates a BoxSampler to sample positive and negative boxes."""

  def __init__(self,
               num_samples=512,
               foreground_fraction=0.25,
               **kwargs):
    """Initializes a box sampler.

    Args:
      num_samples: An `int` of the number of sampled boxes per image.
      foreground_fraction: A `float` in [0, 1], what percentage of boxes should
        be sampled from the positive examples.
      **kwargs: Additional keyword arguments passed to Layer.
    """
    self._config_dict = {
        'num_samples': num_samples,
        'foreground_fraction': foreground_fraction,
    }
    super(BoxSampler, self).__init__(**kwargs)

  def call(self, positive_matches, negative_matches, ignored_matches):
    """Samples and selects positive and negative instances.

    Args:
      positive_matches: A `bool` tensor of shape of [batch, N] where N is the
        number of instances. For each element, `True` means the instance
        corresponds to a positive example.
      negative_matches: A `bool` tensor of shape of [batch, N] where N is the
        number of instances. For each element, `True` means the instance
        corresponds to a negative example.
      ignored_matches: A `bool` tensor of shape of [batch, N] where N is the
        number of instances. For each element, `True` means the instance should
        be ignored.

    Returns:
      A `tf.tensor` of shape of [batch_size, K], storing the indices of the
        sampled examples, where K is `num_samples`.
    """
    sample_candidates = tf.logical_and(
        tf.logical_or(positive_matches, negative_matches),
        tf.logical_not(ignored_matches))

    sampler = sampling_ops.BalancedPositiveNegativeSampler(
        positive_fraction=self._config_dict['foreground_fraction'],
        is_static=True)

    batch_size = sample_candidates.shape[0]
    sampled_indicators = []
    for i in range(batch_size):
      sampled_indicator = sampler.subsample(
          sample_candidates[i],
          self._config_dict['num_samples'],
          positive_matches[i])
      sampled_indicators.append(sampled_indicator)
    sampled_indicators = tf.stack(sampled_indicators)
    _, selected_indices = tf.nn.top_k(
        tf.cast(sampled_indicators, dtype=tf.int32),
        k=self._config_dict['num_samples'],
        sorted=True)

    return selected_indices

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
