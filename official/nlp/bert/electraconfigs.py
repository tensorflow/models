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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import dataclasses
import json
import six
import tensorflow as tf
from official.modeling import hyperparams

@dataclasses.dataclass
class ElectraConfig(hyperparams.Config):
  """Configuration for `ElectraModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=256,
               num_hidden_layers=4,
               num_attention_heads=4,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               multiplier=3,
               discrim_rate=50):
    """Constructs ElectraConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.multiplier = multiplier
    self.discrim_hidden_size = self.hidden_size*self.multiplier
    self.discrim_layers = self.num_hidden_layers*self.multiplier
    self.discrim_rate = discrim_rate
    self.discrim_attention_heads = self.num_attention_heads*self.multiplier
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = ElectraConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  def get_generator_bert(self):
    return configs.BertConfig(self.vocab_size,
      self.hidden_size,
      self.num_hidden_layers,
      self.num_attention_heads,
      self.intermediate_size,
      self.hidden_act,
      self.hidden_dropout_prob,
      self.attention_probs_dropout_prob,
      self.max_position_embeddings,
      self.type_vocab_size,
      self.initializer_range)

  def get_discriminator_bert(self):
    return configs.BertConfig(self.vocab_size,
      self.discrim_hidden_size,
      self.discrim_layers,
      self.discrim_attention_heads,
      self.intermediate_size,
      self.hidden_act,
      self.hidden_dropout_prob,
      self.attention_probs_dropout_prob,
      self.max_position_embeddings,
      self.type_vocab_size,
      self.initializer_range)

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

