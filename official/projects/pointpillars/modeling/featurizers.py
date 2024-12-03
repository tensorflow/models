# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Featurizer layers for Pointpillars."""

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import layers
from official.projects.pointpillars.utils import utils


@tf_keras.utils.register_keras_serializable(package='Vision')
class Featurizer(tf_keras.layers.Layer):
  """The featurizer to convert pillars to a BEV pseudo image.

  The implementation is from the network architecture of PointPillars
  (https://arxiv.org/pdf/1812.05784.pdf). It extract features from pillar
  tensors then scatter them back to bird-eye-view (BEV) image using indices.

  Notations:
    B: batch size
    H: height of the BEV image
    W: width of the BEV image
    P: number of pillars in an example
    N: number of points in a pillar
    D: number of features in a point
    C: channels of the BEV image
  """

  def __init__(
      self,
      image_size: Tuple[int, int],
      pillars_size: Tuple[int, int, int],
      train_batch_size: int,
      eval_batch_size: int,
      num_blocks: int,
      num_channels: int,
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialize the featurizer.

    Args:
      image_size: A [int, int] tuple to define the [H, W] of BEV image.
      pillars_size: A [int, int, int] tuple to define the [P, N, D] of pillars.
      train_batch_size: An `int` training batch size per replica.
      eval_batch_size: An `int` evaluation batch size per replica.
      num_blocks: An `int` number of blocks for extracting features.
      num_channels: An `int` number channels of the BEV image.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        block layers. Default to None.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(Featurizer, self).__init__(**kwargs)

    self._config_dict = {
        'image_size': image_size,
        'pillars_size': pillars_size,
        'train_batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'num_blocks': num_blocks,
        'num_channels': num_channels,
        'kernel_regularizer': kernel_regularizer,
    }
    self._image_shape = [image_size[0], image_size[1], num_channels]

    utils.assert_channels_last()

  def build(self, input_specs: List[tf.TensorShape]):
    """Creates variables for the featurizer."""
    self._blocks = []
    for _ in range(self._config_dict['num_blocks']):
      self._blocks.append(
          layers.ConvBlock(
              filters=self._config_dict['num_channels'],
              kernel_size=1,
              strides=1,
              kernel_regularizer=self._config_dict['kernel_regularizer']))

    # These batch_dims are [B, P, 1] tensors that could be created before
    # call(). They will be used for tf.scatter_nd to convert pillars to BEV
    # images. Because tf.scatter_nd requires a concrete batch size, we need to
    # prepare all possibilities of batch size for train, eval and test mode.
    self._train_batch_dims = self._get_batch_dims(
        self._config_dict['train_batch_size'])
    self._eval_batch_dims = self._get_batch_dims(
        self._config_dict['eval_batch_size'])
    self._test_batch_dims = self._get_batch_dims(1)

    super(Featurizer, self).build(input_specs)

  def _get_batch_dims(self, batch_size: int) -> tf.Tensor:
    p = self._config_dict['pillars_size'][0]
    batch_dims = np.indices([batch_size, p])[0]
    batch_dims = tf.convert_to_tensor(batch_dims, dtype=tf.int32)
    batch_dims = tf.expand_dims(batch_dims, axis=-1)
    return batch_dims

  def _get_batch_size_and_dims(self,  # pytype: disable=annotation-type-mismatch
                               training: bool = None) -> Tuple[int, tf.Tensor]:
    # We use training as a ternary indicator, None for test mode.
    # Test mode will be used for saving model and model inference.
    if training is None:
      batch_size = 1
      batch_dims = self._test_batch_dims
    else:
      if training:
        batch_size = self._config_dict['train_batch_size']
        batch_dims = self._train_batch_dims
      else:
        batch_size = self._config_dict['eval_batch_size']
        batch_dims = self._eval_batch_dims
    return batch_size, batch_dims

  def call(self,  # pytype: disable=annotation-type-mismatch
           pillars: tf.Tensor,
           indices: tf.Tensor,
           training: bool = None) -> tf.Tensor:
    """Forward pass of the featurizer."""
    # Add batch index to pillar indices.
    # (B, P, 1)
    batch_size, batch_dims = self._get_batch_size_and_dims(training)
    # (B, P, 3)
    batch_indices = tf.concat([batch_dims, indices], axis=-1)

    # Extract features from pillars.
    # (B, P, N, D)
    x = pillars
    # (B, P, N, C)
    for block in self._blocks:
      x = block(x)
    # (B, P, C)
    x = tf.reduce_max(x, axis=2, keepdims=False)

    # Scatter pillars back to form a BEV image.
    # (B, H, W, C)
    image = tf.scatter_nd(
        batch_indices,
        x,
        shape=[batch_size] + self._image_shape)
    self._output_specs = image.get_shape()
    return image

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> tf_keras.Model:
    return cls(**config)

  @property
  def output_specs(self) -> tf.TensorShape:
    return self._output_specs
