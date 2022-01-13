# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""TEAMS model configurations and instantiation methods."""
import dataclasses

import gin
import tensorflow as tf

from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.modeling import layers
from official.nlp.modeling import networks


@dataclasses.dataclass
class TeamsPretrainerConfig(base_config.Config):
  """Teams pretrainer configuration."""
  # Candidate size for multi-word selection task, including the correct word.
  candidate_size: int = 5
  # Weight for the generator masked language model task.
  generator_loss_weight: float = 1.0
  # Weight for the replaced token detection task.
  discriminator_rtd_loss_weight: float = 5.0
  # Weight for the multi-word selection task.
  discriminator_mws_loss_weight: float = 2.0
  # Whether share embedding network between generator and discriminator.
  tie_embeddings: bool = True
  # Number of bottom layers shared between generator and discriminator.
  # Non-positive value implies no sharing.
  num_shared_generator_hidden_layers: int = 3
  # Number of bottom layers shared between different discriminator tasks.
  num_discriminator_task_agnostic_layers: int = 11
  generator: encoders.BertEncoderConfig = encoders.BertEncoderConfig()
  discriminator: encoders.BertEncoderConfig = encoders.BertEncoderConfig()


class TeamsEncoderConfig(encoders.BertEncoderConfig):
  pass


@gin.configurable
@base_config.bind(TeamsEncoderConfig)
def get_encoder(bert_config: TeamsEncoderConfig,
                embedding_network=None,
                hidden_layers=None):
  """Gets a 'EncoderScaffold' object.

  Args:
    bert_config: A 'modeling.BertConfig'.
    embedding_network: Embedding network instance.
    hidden_layers: List of hidden layer instances.

  Returns:
    A encoder object.
  """
  embedding_cfg = dict(
      vocab_size=bert_config.vocab_size,
      type_vocab_size=bert_config.type_vocab_size,
      hidden_size=bert_config.hidden_size,
      embedding_width=bert_config.embedding_size,
      max_seq_length=bert_config.max_position_embeddings,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      dropout_rate=bert_config.dropout_rate,
  )
  hidden_cfg = dict(
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      intermediate_activation=tf_utils.get_activation(
          bert_config.hidden_activation),
      dropout_rate=bert_config.dropout_rate,
      attention_dropout_rate=bert_config.attention_dropout_rate,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
  )
  if embedding_network is None:
    embedding_network = networks.PackedSequenceEmbedding
  if hidden_layers is None:
    hidden_layers = layers.Transformer
  kwargs = dict(
      embedding_cfg=embedding_cfg,
      embedding_cls=embedding_network,
      hidden_cls=hidden_layers,
      hidden_cfg=hidden_cfg,
      num_hidden_instances=bert_config.num_layers,
      pooled_output_dim=bert_config.hidden_size,
      pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      dict_outputs=True)

  # Relies on gin configuration to define the Transformer encoder arguments.
  return networks.EncoderScaffold(**kwargs)
