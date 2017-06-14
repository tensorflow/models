# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Entropy coder model."""


class EntropyCoderModel(object):
  """Entropy coder model."""

  def __init__(self):
    # Loss used for training the model.
    self.loss = None

    # Tensorflow op to run to train the model.
    self.train_op = None

    # Tensor corresponding to the average code length of the input bit field
    # tensor. The average code length is a number of output bits per input bit.
    # To get an effective compression, this number should be between 0.0
    # and 1.0 (1.0 corresponds to no compression).
    self.average_code_length = None

  def Initialize(self, global_step, optimizer, config_string):
    raise NotImplementedError()

  def BuildGraph(self, input_codes):
    """Build the Tensorflow graph corresponding to the entropy coder model.

    Args:
      input_codes: Tensor of size: batch_size x height x width x bit_depth
        corresponding to the codes to compress.
        The input codes are {-1, +1} codes.
    """
    # TODO:
    # - consider switching to {0, 1} codes.
    # - consider passing an extra tensor which gives for each (b, y, x)
    #   what is the actual depth (which would allow to use more or less bits
    #   for each (y, x) location.
    raise NotImplementedError()

  def GetConfigStringForUnitTest(self):
    """Returns a default model configuration to be used for unit tests."""
    return None
