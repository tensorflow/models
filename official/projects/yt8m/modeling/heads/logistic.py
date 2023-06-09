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

"""Logistic model definitions."""

from typing import Optional

import tensorflow as tf


layers = tf.keras.layers


class LogisticModel(tf.keras.Model):
  """Logistic prediction head model with L2 regularization."""

  def __init__(
      self,
      input_specs: layers.InputSpec = layers.InputSpec(shape=[None, 128]),
      vocab_size: int = 3862,
      return_logits: bool = False,
      l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs,
  ):
    """Creates a logistic model.

    Args:
      input_specs: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      return_logits: if True also return logits.
      l2_regularizer: An optional L2 weight regularizer.
      **kwargs: extra key word args.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    logits = layers.Dense(vocab_size, kernel_regularizer=l2_regularizer)(inputs)

    outputs = {"predictions": tf.nn.sigmoid(logits)}
    if return_logits:
      outputs.update({"logits": logits})

    super().__init__(inputs=inputs, outputs=outputs, **kwargs)
