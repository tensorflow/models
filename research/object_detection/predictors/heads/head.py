# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Base head class.

All the different kinds of prediction heads in different models will inherit
from this class. What is in common between all head classes is that they have a
`predict` function that receives `features` as its first argument.

How to add a new prediction head to an existing meta architecture?
For example, how can we add a `3d shape` prediction head to Mask RCNN?

We have to take the following steps to add a new prediction head to an
existing meta arch:
(a) Add a class for predicting the head. This class should inherit from the
`Head` class below and have a `predict` function that receives the features
and predicts the output. The output is always a tf.float32 tensor.
(b) Add the head to the meta architecture. For example in case of Mask RCNN,
go to box_predictor_builder and put in the logic for adding the new head to the
Mask RCNN box predictor.
(c) Add the logic for computing the loss for the new head.
(d) Add the necessary metrics for the new head.
(e) (optional) Add visualization for the new head.
"""
from abc import abstractmethod

import tensorflow as tf


class Head(object):
  """Mask RCNN head base class."""

  def __init__(self):
    """Constructor."""
    pass

  @abstractmethod
  def predict(self, features, num_predictions_per_location):
    """Returns the head's predictions.

    Args:
      features: A float tensor of features.
      num_predictions_per_location: Int containing number of predictions per
        location.

    Returns:
      A tf.float32 tensor.
    """
    pass


class KerasHead(tf.keras.Model):
  """Keras head base class."""

  def call(self, features):
    """The Keras model call will delegate to the `_predict` method."""
    return self._predict(features)

  @abstractmethod
  def _predict(self, features):
    """Returns the head's predictions.

    Args:
      features: A float tensor of features.

    Returns:
      A tf.float32 tensor.
    """
    pass
