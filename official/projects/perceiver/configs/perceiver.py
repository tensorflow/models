# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Perceiver configurations."""

import dataclasses

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.nlp.data import pretrain_dataloader
from official.nlp.data import sentence_prediction_dataloader


_SENTENCE_PREDICTION_TRAINER = cfg.TrainerConfig(
    optimizer_config=optimization.OptimizationConfig({
        'optimizer': {
            'type': 'lamb',
            'lamb': {
                'weight_decay_rate': 0.01,
                'exclude_from_weight_decay': [
                    'LayerNorm', 'layer_norm', 'bias'
                ],
            }
        },
        'learning_rate': {
            'type': 'polynomial',
            'polynomial': {
                'initial_learning_rate': 3.0e-05,
                'end_learning_rate': 0.0,
                'decay_steps': 32730,
                'power': 1.0,
            }
        },
        'warmup': {
            'type': 'linear',
            'linear': {
                'warmup_steps': 200,
                'warmup_learning_rate': 0.,
            }
        }
    }))

_MLM_WORDPIECE_TRAINER = cfg.TrainerConfig(
    train_steps=500_000,
    optimizer_config=optimization.OptimizationConfig({
        'optimizer': {
            'type': 'lamb',
            'lamb': {
                'weight_decay_rate': 0.01,
                'exclude_from_weight_decay': [
                    'LayerNorm', 'layer_norm', 'bias'
                ],
            }
        },
        'learning_rate': {
            'type': 'cosine',
            'cosine': {
                'initial_learning_rate': 1.25e-3,
                'decay_steps': 500_000,
            }
        },
        'warmup': {
            'type': 'linear',
            'linear': {
                'warmup_steps': 1_000,
                'warmup_learning_rate': 0.,
            }
        }
    }))


@dataclasses.dataclass
class EncoderConfig(base_config.Config):
  """The perceiver encoder processor configuration."""

  _attention_heads = 8
  _per_attention_head_last_dim = 32
  self_attention_widening_factor: int = 1
  self_attention_num_heads: int = _attention_heads
  cross_attention_widening_factor: int = 1
  cross_attention_num_heads: int = _attention_heads
  num_self_attends_per_block: int = 26
  num_blocks: int = 1
  qk_last_dim: int = _attention_heads * _per_attention_head_last_dim
  v_last_dim: int = 1280
  dropout_prob: float = 0.0
  dropout_attn_prob: float = 0.0
  att_init_scale: float = 1.0
  dense_init_scale: float = 1.0
  norm_epsilon: float = 1e-5


@dataclasses.dataclass
class DecoderConfig(base_config.Config):
  """The perceiver decoder configuration."""
  num_heads: int = 8
  _per_attention_head_last_dim = 32
  output_last_dim: int = 768
  qk_last_dim: int = num_heads * _per_attention_head_last_dim
  v_last_dim: int = 768
  use_query_residual: bool = False


@dataclasses.dataclass
class PositionalDecoder(base_config.Config):
  d_model: int = 768
  decoder: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)
  position_encoding_intializer_stddev: float = 0.02
  output_index_dim: int = 512
  d_latents: int = 1280
  z_index_dim: int = 256


@dataclasses.dataclass
class ClassificationDecoderConfig(PositionalDecoder):
  output_index_dim: int = 1


@dataclasses.dataclass
class MaskedLMDecoderConfig(PositionalDecoder):
  output_index_dim: int = 512


@dataclasses.dataclass
class SequenceEncoderConfig(base_config.Config):
  """The perceiver sequence encoder configuration."""
  d_model: int = 768
  d_latents: int = 1280
  z_index_dim: int = 256
  max_seq_len: int = 512
  vocab_size: int = 30_522
  embedding_width: int = 768
  embedding_initializer_stddev: float = 0.02
  input_position_encoding_intializer_stddev: float = 0.02
  z_pos_enc_init_scale: float = 0.02

  encoder: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)


@dataclasses.dataclass
class PretrainerConfig(base_config.Config):
  """The pretrainer configuration."""
  encoder: SequenceEncoderConfig = dataclasses.field(
      default_factory=SequenceEncoderConfig
  )
  decoder: MaskedLMDecoderConfig = dataclasses.field(
      default_factory=MaskedLMDecoderConfig
  )

  mlm_activation: str = 'gelu'
  mlm_initializer_range: float = 0.02


@dataclasses.dataclass
class ClassificationConfig(base_config.Config):
  """The classification configuration."""
  num_classes: int = 0
  use_encoder_pooler: bool = False
  encoder: SequenceEncoderConfig = dataclasses.field(
      default_factory=SequenceEncoderConfig
  )
  decoder: ClassificationDecoderConfig = dataclasses.field(
      default_factory=ClassificationDecoderConfig
  )


@dataclasses.dataclass
class SentencePredictionConfig(cfg.TaskConfig):
  """The sentence prediction task config."""

  model: ClassificationConfig = dataclasses.field(
      default_factory=ClassificationConfig
  )

  hub_module_url: str = ''
  init_checkpoint: str = ''
  init_cls_pooler: bool = False

  metric_type: str = 'accuracy'

  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )


@dataclasses.dataclass
class PretrainConfig(cfg.TaskConfig):
  """The word piece pretrain task config."""

  model: PretrainerConfig = dataclasses.field(default_factory=PretrainerConfig)
  init_checkpoint: str = ''

  scale_loss: bool = False

  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )


@exp_factory.register_config_factory('perceiver/word_piece_sentence_prediction')
def perceiver_word_piece_sentence_prediction() -> cfg.ExperimentConfig:
  """Config for perceiver sentence prediction.

  Returns:
    cfg.ExperimentConfig
  References:
    Perceiver IO (https://arxiv.org/abs/2107.14795).
  """

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=SentencePredictionConfig(
          train_data=sentence_prediction_dataloader
          .SentencePredictionDataConfig(),
          validation_data=sentence_prediction_dataloader
          .SentencePredictionDataConfig()),
      trainer=_SENTENCE_PREDICTION_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory(
    'perceiver/word_piece_raw_sentence_prediction'
)
def perceiver_word_piece_raw_sentence_prediction() -> cfg.ExperimentConfig:
  """Config for perceiver sentence prediction.

  Returns:
    cfg.ExperimentConfig
  References:
    Perceiver IO (https://arxiv.org/abs/2107.14795).
  """

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=SentencePredictionConfig(
          train_data=sentence_prediction_dataloader.SentencePredictionTextDataConfig(),
          validation_data=sentence_prediction_dataloader.SentencePredictionTextDataConfig(),
      ),
      trainer=_SENTENCE_PREDICTION_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ],
  )
  return config


@exp_factory.register_config_factory('perceiver/wordpiece_pretrain')
def perceiver_wordpiece_pretrain() -> cfg.ExperimentConfig:
  """Config for perceiver wordpiece pretrain.

  Returns:
    cfg.ExperimentConfig
  References:
    Perceiver IO (https://arxiv.org/abs/2107.14795).
    Bert pretraining data
    (https://github.com/google-research/bert/blob/master/tokenization.py#L168)
  """

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(enable_xla=True),
      task=PretrainConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(
              global_batch_size=512,
              use_next_sentence_label=False,
              use_v2_feature_names=True),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              global_batch_size=512,
              is_training=False,
              use_next_sentence_label=False,
              use_v2_feature_names=True)),
      trainer=_MLM_WORDPIECE_TRAINER,
      restrictions=[
          'task.train_data.is_training != None',
      ])
  return config
