# Copyright 2023 The Orbit Authors. All Rights Reserved.
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

"""Provides AbstractTrainer/Evaluator base classes, defining train/eval APIs."""

import abc

from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf, tf_keras


Output = Dict[str, Union[tf.Tensor, float, np.number, np.ndarray, 'Output']]  # pytype: disable=not-supported-yet


class AbstractTrainer(tf.Module, metaclass=abc.ABCMeta):
  """An abstract class defining the API required for training."""

  @abc.abstractmethod
  def train(self, num_steps: tf.Tensor) -> Optional[Output]:
    """Implements `num_steps` steps of training.

    This method will be called by the `Controller` to perform the "inner loop"
    of training. This inner loop amortizes the cost of bookkeeping associated
    with checkpointing, evaluation, and writing summaries. Additionally, the
    inner loop can be implemented (if desired) using TensorFlow's looping
    constructs (e.g. a `for` loop over a `tf.range` inside a `tf.function`),
    which can be necessary for getting optimal performance when running on TPU.
    For cases that don't require peak performance, a simple Python loop can be
    used instead for simplicity.

    Args:
      num_steps: The number of training steps to run. Note that it is up to the
        model what constitutes a "step", which may involve more than one update
        to model parameters (e.g., if training a GAN).

    Returns:
      Either `None`, or a dictionary mapping names to `Tensor`s or NumPy values.
      If a dictionary is returned, it will be written to logs and as TensorBoard
      summaries. The dictionary may also be nested, which will generate a
      hierarchy of summary directories.
    """
    pass


class AbstractEvaluator(tf.Module, metaclass=abc.ABCMeta):
  """An abstract class defining the API required for evaluation."""

  @abc.abstractmethod
  def evaluate(self, num_steps: tf.Tensor) -> Optional[Output]:
    """Implements `num_steps` steps of evaluation.

    This method will by called the `Controller` to perform an evaluation. The
    `num_steps` parameter specifies the number of steps of evaluation to run,
    which is specified by the user when calling one of the `Controller`'s
    evaluation methods. A special sentinel value of `-1` is reserved to indicate
    evaluation should run until the underlying data source is exhausted.

    Args:
      num_steps: The number of evaluation steps to run. Note that it is up to
        the model what constitutes a "step". Evaluations may also want to
        support "complete" evaluations when `num_steps == -1`, running until a
        given data source is exhausted.

    Returns:
      Either `None`, or a dictionary mapping names to `Tensor`s or NumPy values.
      If a dictionary is returned, it will be written to logs and as TensorBoard
      summaries. The dictionary may also be nested, which will generate a
      hierarchy of summary directories.
    """
    pass
