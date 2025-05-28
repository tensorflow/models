# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Defines base abstract uplift network layers."""

import abc

import tensorflow as tf, tf_keras

from official.recommendation.uplift import types


class BaseTwoTowerUpliftNetwork(tf_keras.layers.Layer, metaclass=abc.ABCMeta):
  """Abstract class for uplift layers that compute control and treatment logits.

  A TwoTowerUpliftNetwork layer is expected to take in a dictionary of feature
  tensors and compute two sets of logits: one for the control group and one for
  treatment group.
  """

  @abc.abstractmethod
  def call(
      self,
      inputs: types.DictOfTensors,
      training: bool | None = None,
      mask: tf.Tensor | None = None,
  ) -> types.TwoTowerTrainingOutputs:
    raise NotImplementedError()
