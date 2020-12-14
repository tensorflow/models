# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""The ALBERT configurations."""

import six

from official.nlp.bert import configs


class AlbertConfig(configs.BertConfig):
  """Configuration for `ALBERT`."""

  def __init__(self, num_hidden_groups=1, inner_group_num=1, **kwargs):
    """Constructs AlbertConfig.

    Args:
      num_hidden_groups: Number of group for the hidden layers, parameters in
        the same group are shared. Note that this value and also the following
        'inner_group_num' has to be 1 for now, because all released ALBERT
        models set them to 1. We may support arbitary valid values in future.
      inner_group_num: Number of inner repetition of attention and ffn.
      **kwargs: The remaining arguments are the same as above 'BertConfig'.
    """
    super(AlbertConfig, self).__init__(**kwargs)

    # TODO(chendouble): 'inner_group_num' and 'num_hidden_groups' are always 1
    # in the released ALBERT. Support other values in AlbertEncoder if needed.
    if inner_group_num != 1 or num_hidden_groups != 1:
      raise ValueError("We only support 'inner_group_num' and "
                       "'num_hidden_groups' as 1.")

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `AlbertConfig` from a Python dictionary of parameters."""
    config = AlbertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config
