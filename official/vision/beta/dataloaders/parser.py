# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""The generic parser interface."""

import abc


class Parser(object):
  """Parses data and produces tensors to be consumed by models."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def _parse_train_data(self, decoded_tensors):
    """Generates images and labels that are usable for model training.

    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.

    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    pass

  @abc.abstractmethod
  def _parse_eval_data(self, decoded_tensors):
    """Generates images and labels that are usable for model evaluation.

    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.

    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    pass

  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.

    Args:
      is_training: a `bool` to indicate whether it is in training mode.

    Returns:
      parse: a `callable` that takes the serialized example and generate the
        images, labels tuple where labels is a dict of Tensors that contains
        labels.
    """
    def parse(decoded_tensors):
      """Parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse

  @classmethod
  def inference_fn(cls, inputs):
    """Parses inputs for predictions.

    Args:
      inputs: A Tensor, or dictionary of Tensors.

    Returns:
      processed_inputs: An input tensor to the model.
    """
    pass
