# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Layer config for parsing ODAPI checkpoint.

This file contains the layers (Config objects) that are used for parsing the
ODAPI checkpoint weights for CenterNet.

Currently, the parser is incomplete and has only been tested on
CenterNet Hourglass-104 512x512.
"""

import abc
import dataclasses
from typing import Dict, Optional

import numpy as np
import tensorflow as tf


class Config(abc.ABC):
  """Base config class."""

  def get_weights(self):
    """Generates the weights needed to be loaded into the layer."""
    raise NotImplementedError

  def load_weights(self, layer: tf.keras.layers.Layer) -> int:
    """Assign weights to layer.

    Given a layer, this function retrieves the weights for that layer in an
    appropriate format and order, and loads them into the layer. Additionally,
    the number of weights loaded are returned.

    If the weights are in an incorrect format, a ValueError
    will be raised by set_weights().

    Args:
      layer: A `tf.keras.layers.Layer`.

    Returns:

    """
    weights = self.get_weights()
    layer.set_weights(weights)

    n_weights = 0
    for w in weights:
      n_weights += w.size
    return n_weights


@dataclasses.dataclass
class Conv2DBNCFG(Config):
  """Config class for Conv2DBN block."""

  weights_dict: Optional[Dict[str, np.ndarray]] = dataclasses.field(
      repr=False, default=None)

  weights: Optional[np.ndarray] = dataclasses.field(repr=False, default=None)
  beta: Optional[np.ndarray] = dataclasses.field(repr=False, default=None)
  gamma: Optional[np.ndarray] = dataclasses.field(repr=False, default=None)
  moving_mean: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  moving_variance: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)

  def __post_init__(self):
    conv_weights_dict = self.weights_dict['conv']
    norm_weights_dict = self.weights_dict['norm']

    self.weights = conv_weights_dict['kernel']

    self.beta = norm_weights_dict['beta']
    self.gamma = norm_weights_dict['gamma']
    self.moving_mean = norm_weights_dict['moving_mean']
    self.moving_variance = norm_weights_dict['moving_variance']

  def get_weights(self):
    return [
        self.weights,
        self.gamma,
        self.beta,
        self.moving_mean,
        self.moving_variance
    ]


@dataclasses.dataclass
class ResidualBlockCFG(Config):
  """Config class for Residual block."""

  weights_dict: Optional[Dict[str, np.ndarray]] = dataclasses.field(
      repr=False, default=None)

  skip_weights: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  skip_beta: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  skip_gamma: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  skip_moving_mean: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  skip_moving_variance: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)

  conv_weights: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  norm_beta: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  norm_gamma: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  norm_moving_mean: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  norm_moving_variance: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)

  conv_block_weights: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  conv_block_beta: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  conv_block_gamma: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  conv_block_moving_mean: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  conv_block_moving_variance: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)

  def __post_init__(self):
    conv_weights_dict = self.weights_dict['conv']
    norm_weights_dict = self.weights_dict['norm']
    conv_block_weights_dict = self.weights_dict['conv_block']

    if 'skip' in self.weights_dict:
      skip_weights_dict = self.weights_dict['skip']
      self.skip_weights = skip_weights_dict['conv']['kernel']
      self.skip_beta = skip_weights_dict['norm']['beta']
      self.skip_gamma = skip_weights_dict['norm']['gamma']
      self.skip_moving_mean = skip_weights_dict['norm']['moving_mean']
      self.skip_moving_variance = skip_weights_dict['norm']['moving_variance']

    self.conv_weights = conv_weights_dict['kernel']
    self.norm_beta = norm_weights_dict['beta']
    self.norm_gamma = norm_weights_dict['gamma']
    self.norm_moving_mean = norm_weights_dict['moving_mean']
    self.norm_moving_variance = norm_weights_dict['moving_variance']

    self.conv_block_weights = conv_block_weights_dict['conv']['kernel']
    self.conv_block_beta = conv_block_weights_dict['norm']['beta']
    self.conv_block_gamma = conv_block_weights_dict['norm']['gamma']
    self.conv_block_moving_mean = conv_block_weights_dict['norm']['moving_mean']
    self.conv_block_moving_variance = conv_block_weights_dict['norm'][
        'moving_variance']

  def get_weights(self):
    weights = [
        self.skip_weights,
        self.skip_gamma,
        self.skip_beta,

        self.conv_block_weights,
        self.conv_block_gamma,
        self.conv_block_beta,

        self.conv_weights,
        self.norm_gamma,
        self.norm_beta,

        self.skip_moving_mean,
        self.skip_moving_variance,
        self.conv_block_moving_mean,
        self.conv_block_moving_variance,
        self.norm_moving_mean,
        self.norm_moving_variance,
    ]

    weights = [x for x in weights if x is not None]
    return weights


@dataclasses.dataclass
class HeadConvCFG(Config):
  """Config class for HeadConv block."""

  weights_dict: Optional[Dict[str, np.ndarray]] = dataclasses.field(
      repr=False, default=None)

  conv_1_weights: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  conv_1_bias: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)

  conv_2_weights: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)
  conv_2_bias: Optional[np.ndarray] = dataclasses.field(
      repr=False, default=None)

  def __post_init__(self):
    conv_1_weights_dict = self.weights_dict['layer_with_weights-0']
    conv_2_weights_dict = self.weights_dict['layer_with_weights-1']

    self.conv_1_weights = conv_1_weights_dict['kernel']
    self.conv_1_bias = conv_1_weights_dict['bias']
    self.conv_2_weights = conv_2_weights_dict['kernel']
    self.conv_2_bias = conv_2_weights_dict['bias']

  def get_weights(self):
    return [
        self.conv_1_weights,
        self.conv_1_bias,
        self.conv_2_weights,
        self.conv_2_bias
    ]


@dataclasses.dataclass
class HourglassCFG(Config):
  """Config class for Hourglass block."""

  weights_dict: Optional[Dict[str, np.ndarray]] = dataclasses.field(
      repr=False, default=None)
  is_last_stage: bool = dataclasses.field(repr=False, default=None)

  def __post_init__(self):
    self.is_last_stage = False if 'inner_block' in self.weights_dict else True

  def get_weights(self):
    """It is not used in this class."""
    return None

  def generate_block_weights(self, weights_dict):
    """Convert weights dict to blocks structure."""

    reps = len(weights_dict.keys())
    weights = []
    n_weights = 0

    for i in range(reps):
      res_config = ResidualBlockCFG(weights_dict=weights_dict[str(i)])
      res_weights = res_config.get_weights()
      weights += res_weights

      for w in res_weights:
        n_weights += w.size

    return weights, n_weights

  def load_block_weights(self, layer, weight_dict):
    block_weights, n_weights = self.generate_block_weights(weight_dict)
    layer.set_weights(block_weights)
    return n_weights

  def load_weights(self, layer):
    n_weights = 0

    if not self.is_last_stage:
      enc_dec_layers = [
          layer.submodules[0],
          layer.submodules[1],
          layer.submodules[3]
      ]
      enc_dec_weight_dicts = [
          self.weights_dict['encoder_block1'],
          self.weights_dict['encoder_block2'],
          self.weights_dict['decoder_block']
      ]

      for l, weights_dict in zip(enc_dec_layers, enc_dec_weight_dicts):
        n_weights += self.load_block_weights(l, weights_dict)

      if len(self.weights_dict['inner_block']) == 1:
        # still in an outer hourglass
        inner_weights_dict = self.weights_dict['inner_block']['0']
      else:
        # inner residual block chain
        inner_weights_dict = self.weights_dict['inner_block']

      inner_hg_layer = layer.submodules[2]
      inner_hg_cfg = type(self)(weights_dict=inner_weights_dict)
      n_weights += inner_hg_cfg.load_weights(inner_hg_layer)

    else:
      inner_layer = layer.submodules[0]
      n_weights += self.load_block_weights(inner_layer, self.weights_dict)

    return n_weights
