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

"""Logistic model definitions."""

from typing import Optional

import tensorflow as tf, tf_keras


layers = tf_keras.layers


class LogisticModel(layers.Layer):
  """Logistic prediction head model with L2 regularization."""

  def __init__(
      self,
      vocab_size: int = 3862,
      return_logits: bool = False,
      l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs,
  ):
    """Creates a logistic model.

    Args:
      vocab_size: The number of classes in the dataset.
      return_logits: if True also return logits.
      l2_regularizer: An optional L2 weight regularizer.
      **kwargs: extra key word args.
    """
    super().__init__(**kwargs)
    self._return_logits = return_logits
    self._dense = layers.Dense(vocab_size, kernel_regularizer=l2_regularizer)

  def call(
      self,
      inputs: tf.Tensor,
  ):
    """Logistic model forward call.

    Args:
      inputs: 'batch' x 'num_features' matrix of input features.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """

    logits = self._dense(inputs)
    outputs = {"predictions": tf.nn.sigmoid(logits)}
    if self._return_logits:
      outputs.update({"logits": logits})
    return outputs
