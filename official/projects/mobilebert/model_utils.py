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

"""Checkpoint converter for Mobilebert."""
import copy
import json

import tensorflow.compat.v1 as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling import models
from official.nlp.modeling import networks


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02,
               embedding_size=None,
               trigram_input=False,
               use_bottleneck=False,
               intra_bottleneck_size=None,
               use_bottleneck_attention=False,
               key_query_shared_bottleneck=False,
               num_feedforward_networks=1,
               normalization_type="layer_norm",
               classifier_activation=True):
    """Constructs BertConfig.

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
      embedding_size: The size of the token embedding.
      trigram_input: Use a convolution of trigram as input.
      use_bottleneck: Use the bottleneck/inverted-bottleneck structure in BERT.
      intra_bottleneck_size: The hidden size in the bottleneck.
      use_bottleneck_attention: Use attention inputs from the bottleneck
        transformation.
      key_query_shared_bottleneck: Use the same linear transformation for
        query&key in the bottleneck.
      num_feedforward_networks: Number of FFNs in a block.
      normalization_type: The normalization type in BERT.
      classifier_activation: Using the tanh activation for the final
        representation of the [CLS] token in fine-tuning.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.embedding_size = embedding_size
    self.trigram_input = trigram_input
    self.use_bottleneck = use_bottleneck
    self.intra_bottleneck_size = intra_bottleneck_size
    self.use_bottleneck_attention = use_bottleneck_attention
    self.key_query_shared_bottleneck = key_query_shared_bottleneck
    self.num_feedforward_networks = num_feedforward_networks
    self.normalization_type = normalization_type
    self.classifier_activation = classifier_activation

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in json_object.items():
      config.__dict__[key] = value
    if config.embedding_size is None:
      config.embedding_size = config.hidden_size
    if config.intra_bottleneck_size is None:
      config.intra_bottleneck_size = config.hidden_size
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def create_mobilebert_pretrainer(bert_config):
  """Creates a BertPretrainerV2 that wraps MobileBERTEncoder model."""
  mobilebert_encoder = networks.MobileBERTEncoder(
      word_vocab_size=bert_config.vocab_size,
      word_embed_size=bert_config.embedding_size,
      type_vocab_size=bert_config.type_vocab_size,
      max_sequence_length=bert_config.max_position_embeddings,
      num_blocks=bert_config.num_hidden_layers,
      hidden_size=bert_config.hidden_size,
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      intermediate_act_fn=tf_utils.get_activation(bert_config.hidden_act),
      hidden_dropout_prob=bert_config.hidden_dropout_prob,
      attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
      intra_bottleneck_size=bert_config.intra_bottleneck_size,
      initializer_range=bert_config.initializer_range,
      use_bottleneck_attention=bert_config.use_bottleneck_attention,
      key_query_shared_bottleneck=bert_config.key_query_shared_bottleneck,
      num_feedforward_networks=bert_config.num_feedforward_networks,
      normalization_type=bert_config.normalization_type,
      classifier_activation=bert_config.classifier_activation)

  masked_lm = layers.MobileBertMaskedLM(
      embedding_table=mobilebert_encoder.get_embedding_table(),
      activation=tf_utils.get_activation(bert_config.hidden_act),
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      name="cls/predictions")

  pretrainer = models.BertPretrainerV2(
      encoder_network=mobilebert_encoder, customized_masked_lm=masked_lm)
  # Makes sure the pretrainer variables are created.
  _ = pretrainer(pretrainer.inputs)
  return pretrainer
