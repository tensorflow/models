# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""An abstraction that NLP models define input pipelines."""

import abc
from typing import Optional

import tensorflow as tf


class DataLoader(metaclass=abc.ABCMeta):
  """An abstract class defining the APIs for tf.data input pipeline."""

  @abc.abstractmethod
  def load(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Implements DataLoader load method.

    Builds the entire input pipeline inside the load method. Users can define
    states inside the DataLoader class and returns a tf.data dataset
    object.

    Args:
      input_context: This is a context class that is passed to the user's input
        function and contains information about the compute replicas and input
        pipelines. This object is used for multi-host inputs and passed by the
        distribution strategy.

    Returns:
      A per-host tf.data dataset. Note that, we usually create the distributed
        dataset through the load method, so we should not directly return a
        distributed dataset here.
    """
    pass
