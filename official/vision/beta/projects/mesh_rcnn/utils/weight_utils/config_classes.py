"""Config Classes for weight-loading Mesh R-CNN"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import tensorflow as tf


class Config(ABC):
  def get_weights(self):
    """
    This function generates the weights needed to be loaded into the layer.
    """
    return None

  def load_weights(self, layer: tf.keras.layers.Layer) -> int:
    """
    Given a layer, this function retrives the weights for that layer in an
    appropriate format, and loads them into the layer. Additionally,
    the number of weights loaded are returned. If the weights are in an
    incorrect format, a ValueError will be raised by set_weights().
    """
    weights = self.get_weights()
    layer.set_weights(weights)
    n_weights = 0

    for w in weights:
      n_weights += tf.size(w)
    return n_weights

@dataclass
class meshRefinementStageCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)
  bottleneck_weights: np.array = field(repr=False, default=None)
  bottleneck_bias: np.array = field(repr=False, default=None)
  verts_offset_weights: np.array = field(repr=False, default=None)
  verts_offset_bias: np.array = field(repr=False, default=None)
  gconvs_weights: np.array = field(repr=False, default=None)

  weights: np.array = field(repr=False, default=None)

  def __post_init__(self):
    bottleneck_weights = self.weights_dict['bottleneck']['weight']
    bottleneck_bias = self.weights_dict['bottleneck']['bias']
    verts_offset_weights = self.weights_dict['verts_offset']['weight']
    verts_offset_bias = self.weights_dict['verts_offset']['bias']

    self.weights = [
        bottleneck_weights,
        bottleneck_bias,
        verts_offset_weights,
        verts_offset_bias
    ]

    for stage in self.weights_dict['gconvs'].keys():
      gconvConfig = graphConvCFG(
          weights_dict=self.weights_dict['gconvs'][stage])
      self.weights += gconvConfig.get_weights()

  def get_weights(self):
    return self.weights

@dataclass
class graphConvCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)
  w0_weights: np.array = field(repr=False, default=None)
  w0_bias: np.array = field(repr=False, default=None)
  w1_weights: np.array = field(repr=False, default=None)
  w1_bias: np.array = field(repr=False, default=None)

  def __post_init__(self):
    self.w0_weights = self.weights_dict['w0']['weight']
    self.w0_bias = self.weights_dict['w0']['bias']
    self.w1_weights = self.weights_dict['w1']['weight']
    self.w1_bias = self.weights_dict['w1']['bias']

    self.weights = [
        self.w0_weights,
        self.w0_bias,
        self.w1_weights,
        self.w1_bias
    ]

  def get_weights(self):
    return self.weights
