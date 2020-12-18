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
"""Translation task with progressive training."""

from typing import List
# Import libraries
from absl import logging
import dataclasses
import orbit
import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.modeling.progressive import policies
from official.nlp.modeling import models
from official.nlp.tasks import translation


@dataclasses.dataclass
class StackingStageConfig(base_config.Config):
  num_encoder_layers: int = 0
  num_decoder_layers: int = 0
  num_steps: int = 0
  warmup_steps: int = 10000
  initial_learning_rate: float = 0.0625
  power: float = -0.5


@dataclasses.dataclass
class ProgTranslationConfig(translation.TranslationConfig):
  """The progressive model config."""
  model: translation.ModelConfig = translation.ModelConfig(
      encoder=translation.EncDecoder(
          num_attention_heads=16, intermediate_size=4096),
      decoder=translation.EncDecoder(
          num_attention_heads=16, intermediate_size=4096),
      embedding_width=1024,
      padded_decode=True,
      decode_max_length=100)
  optimizer_config: optimization.OptimizationConfig = (
      optimization.OptimizationConfig({
          'optimizer': {
              'type': 'adam',
              'adam': {
                  'beta_2': 0.997,
                  'epsilon': 1e-9,
              },
          },
          'learning_rate': {
              'type': 'power',
              'power': {
                  'initial_learning_rate': 0.0625,
                  'power': -0.5,
              }
          },
          'warmup': {
              'type': 'linear',
              'linear': {
                  'warmup_steps': 16000,
                  'warmup_learning_rate': 0.0
              }
          }
      }))

  stage_list: List[StackingStageConfig] = dataclasses.field(
      default_factory=lambda: [  # pylint: disable=g-long-lambda
          StackingStageConfig(num_encoder_layers=3,
                              num_decoder_layers=3,
                              num_steps=20000,
                              warmup_steps=5000,
                              initial_learning_rate=0.0625),
          StackingStageConfig(num_encoder_layers=6,
                              num_decoder_layers=6,
                              num_steps=20000,
                              warmup_steps=5000,
                              initial_learning_rate=0.0625),
          StackingStageConfig(num_encoder_layers=12,
                              num_decoder_layers=12,
                              num_steps=100000,
                              warmup_steps=5000,
                              initial_learning_rate=0.0625)])


@task_factory.register_task_cls(ProgTranslationConfig)
class ProgressiveTranslationTask(policies.ProgressivePolicy,
                                 translation.TranslationTask):
  """Masked Language Model that supports progressive training.

  Inherate from the TranslationTask class to build model datasets etc.
  """

  def __init__(self, params: cfg.TaskConfig, logging_dir: str = None):
    translation.TranslationTask.__init__(
        self, params=params, logging_dir=logging_dir)
    self._model_config = params.model
    self._optimizer_config = params.optimizer_config
    self._the_only_train_dataset = None
    self._the_only_eval_dataset = None
    policies.ProgressivePolicy.__init__(self)

  # Override
  def num_stages(self):
    return len(self.task_config.stage_list)

  # Override
  def num_steps(self, stage_id):
    return self.task_config.stage_list[stage_id].num_steps

  # Override
  def get_model(self, stage_id, old_model=None):
    """Build model for each stage."""
    num_encoder_layers = (
        self.task_config.stage_list[stage_id].num_encoder_layers)
    num_decoder_layers = (
        self.task_config.stage_list[stage_id].num_decoder_layers)
    params = self._model_config.replace(
        encoder={'num_layers': num_encoder_layers},
        decoder={'num_layers': num_decoder_layers})
    model = self.build_model(params)

    # Run the model once, to make sure that all layers are built.
    # Otherwise, not all weights will be copied.
    inputs = next(tf.nest.map_structure(
        iter, self.build_inputs(self.task_config.train_data)))
    model(inputs, training=True)

    if stage_id > 0 and old_model is not None:
      logging.info('Stage %d copying weights.', stage_id)
      self._copy_weights_to_new_model(old_model=old_model,
                                      new_model=model)
    return model

  # Override
  def build_model(self, params) -> tf.keras.Model:
    """Creates model architecture."""
    model_cfg = params or self.task_config.model
    encoder_kwargs = model_cfg.encoder.as_dict()
    encoder_layer = models.TransformerEncoder(**encoder_kwargs)
    decoder_kwargs = model_cfg.decoder.as_dict()
    decoder_layer = models.TransformerDecoder(**decoder_kwargs)

    return models.Seq2SeqTransformer(
        vocab_size=self._vocab_size,
        embedding_width=model_cfg.embedding_width,
        dropout_rate=model_cfg.dropout_rate,
        padded_decode=model_cfg.padded_decode,
        decode_max_length=model_cfg.decode_max_length,
        beam_size=model_cfg.beam_size,
        alpha=model_cfg.alpha,
        encoder_layer=encoder_layer,
        decoder_layer=decoder_layer)

  # Override
  def get_optimizer(self, stage_id):
    """Build optimizer for each stage."""
    params = self._optimizer_config.replace(
        warmup={
            'linear':
                {'warmup_steps':
                     self.task_config.stage_list[stage_id].warmup_steps
                 },
        },
        learning_rate={
            'power':
                {'initial_learning_rate':
                     self.task_config.stage_list[stage_id].initial_learning_rate
                 },
        },
    )
    opt_factory = optimization.OptimizerFactory(params)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    return optimizer

  # overrides policies.ProgressivePolicy
  def get_train_dataset(self, stage_id):
    del stage_id
    if self._the_only_train_dataset is None:
      strategy = tf.distribute.get_strategy()
      self._the_only_train_dataset = orbit.utils.make_distributed_dataset(
          strategy,
          self.build_inputs,
          self.task_config.train_data)
    return self._the_only_train_dataset

  # overrides policies.ProgressivePolicy
  def get_eval_dataset(self, stage_id):
    del stage_id
    if self._the_only_eval_dataset is None:
      strategy = tf.distribute.get_strategy()
      self._the_only_eval_dataset = orbit.utils.make_distributed_dataset(
          strategy,
          self.build_inputs,
          self.task_config.validation_data)
    return self._the_only_eval_dataset

  def _copy_weights_to_new_model(self, old_model, new_model):
    """Copy model weights from the previous stage to the next.

    Args:
      old_model: nlp.modeling.models.bert_pretrainer.BertPretrainerV2. Model of
        the previous stage.
      new_model: nlp.modeling.models.bert_pretrainer.BertPretrainerV2. Model of
        the next stage.
    """
    new_model.embedding_lookup.set_weights(
        old_model.embedding_lookup.get_weights())
    new_model.position_embedding.set_weights(
        old_model.position_embedding.get_weights())

    new_model.encoder_layer.output_normalization.set_weights(
        old_model.encoder_layer.output_normalization.get_weights())
    new_model.decoder_layer.output_normalization.set_weights(
        old_model.decoder_layer.output_normalization.get_weights())

    old_layer_group = old_model.encoder_layer.encoder_layers
    new_layer_group = new_model.encoder_layer.encoder_layers
    for new_layer_idx in range(len(new_layer_group)):
      old_layer_idx = new_layer_idx % len(old_layer_group)
      new_layer_group[new_layer_idx].set_weights(
          old_layer_group[old_layer_idx].get_weights())

    old_layer_group = old_model.decoder_layer.decoder_layers
    new_layer_group = new_model.decoder_layer.decoder_layers
    for new_layer_idx in range(len(new_layer_group)):
      old_layer_idx = new_layer_idx % len(old_layer_group)
      new_layer_group[new_layer_idx].set_weights(
          old_layer_group[old_layer_idx].get_weights())
