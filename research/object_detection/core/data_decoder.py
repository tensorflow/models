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

"""Interface for data decoders.

Data decoders decode the input data and return a dictionary of tensors keyed by
the entries in core.reader.Fields.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import six


class DataDecoder(six.with_metaclass(ABCMeta, object)):
  """Interface for data decoders."""

  @abstractmethod
  def decode(self, data):
    """Return a single image and associated labels.

    Args:
      data: a string tensor holding a serialized protocol buffer corresponding
        to data for a single image.

    Returns:
      tensor_dict: a dictionary containing tensors. Possible keys are defined in
          reader.Fields.
    """
    pass
