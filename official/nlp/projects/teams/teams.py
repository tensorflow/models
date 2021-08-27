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
from official.nlp.modeling import networks


@dataclasses.dataclass
class TeamsPretrainerConfig(base_config.Config):
  """Teams pretrainer configuration."""
  num_masked_tokens: int = 76
  sequence_length: int = 512
  num_classes: int = 2
  discriminator_loss_weight: float = 50.0
  # Whether share embedding network between generator and discriminator.
  tie_embeddings: bool = True
  # Number of bottom layers shared between generator and discriminator.
  num_shared_generator_hidden_layers: int = 3
  # Number of bottom layers shared between different discriminator tasks.
  num_discriminator_task_agnostic_layers: int = 11
  disallow_correct: bool = False
  generator: encoders.BertEncoderConfig = encoders.BertEncoderConfig()
  discriminator: encoders.BertEncoderConfig = encoders.BertEncoderConfig()


@gin.configurable
def get_encoder(bert_config,
                encoder_scaffold_cls,
                embedding_inst=None,
                hidden_inst=None):
  """Gets a 'EncoderScaffold' object.

  Args:
    bert_config: A 'modeling.BertConfig'.
    encoder_scaffold_cls: An EncoderScaffold class.
    embedding_inst: Embedding instance.
    hidden_inst: List of hidden layer instances.

  Returns:
    A encoder object.
  """
  if embedding_inst is not None:
    # TODO(hongkuny): evaluate if it is better to put cfg definition in gin.
    embedding_cfg = dict(
        vocab_size=bert_config.vocab_size,
        type_vocab_size=bert_config.type_vocab_size,
        hidden_size=bert_config.hidden_size,
        embedding_width=bert_config.embedding_size,
        max_seq_length=bert_config.max_position_embeddings,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range),
        dropout_rate=bert_config.hidden_dropout_prob,
    )
    embedding_inst = networks.PackedSequenceEmbedding(**embedding_cfg)
  hidden_cfg = dict(
      num_attention_heads=bert_config.num_attention_heads,
      intermediate_size=bert_config.intermediate_size,
      intermediate_activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
  )
  kwargs = dict(
      embedding_cfg=embedding_cfg,
      embedding_cls=embedding_inst,
      hidden_cls=hidden_inst,
      hidden_cfg=hidden_cfg,
      num_hidden_instances=bert_config.num_hidden_layers,
      pooled_output_dim=bert_config.hidden_size,
      pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range))

  # Relies on gin configuration to define the Transformer encoder arguments.
  return encoder_scaffold_cls(**kwargs)
