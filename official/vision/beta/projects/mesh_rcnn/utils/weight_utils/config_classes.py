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
class ZHeadCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)
  weights: List = field(repr=False, default=None)

  fc1_weights: np.array = field(repr=False, default=None)
  fc1_bias: np.array = field(repr=False, default=None)
  fc2_weights: np.array = field(repr=False, default=None)
  fc2_bias: np.array = field(repr=False, default=None)
  pred_weights: np.array = field(repr=False, default=None)
  pred_bias: np.array = field(repr=False, default=None)

  def __post_init__(self):
    self.fc1_weights = self.weights_dict['z_fc1']['weight']
    self.fc1_bias = self.weights_dict['z_fc1']['bias']
    self.fc2_weights = self.weights_dict['z_fc2']['weight']
    self.fc2_bias = self.weights_dict['z_fc2']['bias']
    self.pred_weights = self.weights_dict['z_pred']['weight']
    self.pred_bias = self.weights_dict['z_pred']['bias']

    self.weights = [
        self.fc1_weights,
        self.fc1_bias,
        self.fc2_weights,
        self.fc2_bias,
        self.pred_weights,
        self.pred_bias
    ]

  def get_weights(self):
    return self.weights
