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
"""Masked language task with progressive training."""

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
from official.nlp.tasks import masked_lm


@dataclasses.dataclass
class StackingStageConfig(base_config.Config):
  num_layers: int = 0
  num_steps: int = 0
  warmup_steps: int = 10000
  initial_learning_rate: float = 1e-4
  end_learning_rate: float = 0.0
  decay_steps: int = 1000000


@dataclasses.dataclass
class ProgMaskedLMConfig(masked_lm.MaskedLMConfig):
  """The progressive model config."""
  optimizer_config: optimization.OptimizationConfig = (
      optimization.OptimizationConfig(
          optimizer=optimization.OptimizerConfig(type='adamw'),
          learning_rate=optimization.LrConfig(type='polynomial'),
          warmup=optimization.WarmupConfig(type='polynomial'),
      )
  )
  stage_list: List[StackingStageConfig] = dataclasses.field(
      default_factory=lambda: [  # pylint: disable=g-long-lambda
          StackingStageConfig(num_layers=3,
                              num_steps=112500,
                              warmup_steps=10000,
                              initial_learning_rate=1e-4,
                              end_learning_rate=1e-4,
                              decay_steps=112500),
          StackingStageConfig(num_layers=6,
                              num_steps=112500,
                              warmup_steps=10000,
                              initial_learning_rate=1e-4,
                              end_learning_rate=1e-4,
                              decay_steps=112500),
          StackingStageConfig(num_layers=12,
                              num_steps=450000,
                              warmup_steps=10000,
                              initial_learning_rate=1e-4,
                              end_learning_rate=0.0,
                              decay_steps=450000)])


@task_factory.register_task_cls(ProgMaskedLMConfig)
class ProgressiveMaskedLM(policies.ProgressivePolicy, masked_lm.MaskedLMTask):
  """Masked Language Model that supports progressive training.

  Inherate from the MaskedLmTask class to build model datasets etc.
  """

  def __init__(self, params: cfg.TaskConfig, logging_dir: str = None):
    masked_lm.MaskedLMTask.__init__(
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
    num_layers = self.task_config.stage_list[stage_id].num_layers
    encoder_type = self._model_config.encoder.type
    params = self._model_config.replace(
        encoder={encoder_type: {
            'num_layers': num_layers
        }})
    model = self.build_model(params)

    # Run the model once, to make sure that all layers are built.
    # Otherwise, not all weights will be copied.
    _ = model(model.inputs)

    if stage_id > 0 and old_model is not None:
      logging.info('Stage %d copying weights.', stage_id)
      self._copy_weights_to_new_model(old_model=old_model,
                                      new_model=model)
    return model

  # Override
  def get_optimizer(self, stage_id):
    """Build optimizer for each stage."""
    params = self._optimizer_config.replace(
        learning_rate={
            'polynomial':
                {'decay_steps':
                     self.task_config.stage_list[
                         stage_id].decay_steps,
                 'initial_learning_rate':
                     self.task_config.stage_list[
                         stage_id].initial_learning_rate,
                 'end_learning_rate':
                     self.task_config.stage_list[
                         stage_id].end_learning_rate,
                 'power': 1,
                 'cycle': False,
                 }
        },
        warmup={
            'polynomial':
                {'warmup_steps':
                     self.task_config.stage_list[stage_id].warmup_steps,
                 'power': 1,
                }
        }
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
    # Copy weights of the embedding layers.
    # pylint: disable=protected-access
    # When using `encoder_scaffold`, there may be `_embedding_network`.
    if hasattr(new_model.encoder_network, '_embedding_network') and hasattr(
        old_model.encoder_network, '_embedding_network') and (
            new_model.encoder_network._embedding_network is not None):
      new_model.encoder_network._embedding_network.set_weights(
          old_model.encoder_network._embedding_network.get_weights())
    else:
      new_model.encoder_network._embedding_layer.set_weights(
          old_model.encoder_network._embedding_layer.get_weights())
      new_model.encoder_network._position_embedding_layer.set_weights(
          old_model.encoder_network._position_embedding_layer.get_weights())
      new_model.encoder_network._type_embedding_layer.set_weights(
          old_model.encoder_network._type_embedding_layer.get_weights())
      new_model.encoder_network._embedding_norm_layer.set_weights(
          old_model.encoder_network._embedding_norm_layer.get_weights())
    if hasattr(new_model.encoder_network, '_embedding_projection') and hasattr(
        old_model.encoder_network, '_embedding_projection'):
      if old_model.encoder_network._embedding_projection is not None:
        new_model.encoder_network._embedding_projection.set_weights(
            old_model.encoder_network._embedding_projection.get_weights())
    # pylint: enable=protected-access

    # Copy weights of the transformer layers.
    # The model can be EncoderScaffold or TransformerEncoder.
    if hasattr(old_model.encoder_network, 'hidden_layers'):
      old_layer_group = old_model.encoder_network.hidden_layers
    elif hasattr(old_model.encoder_network, 'transformer_layers'):
      old_layer_group = old_model.encoder_network.transformer_layers
    else:
      raise ValueError('Unrecognized encoder network: {}'.format(
          old_model.encoder_network))
    if hasattr(new_model.encoder_network, 'hidden_layers'):
      new_layer_group = new_model.encoder_network.hidden_layers
    elif hasattr(new_model.encoder_network, 'transformer_layers'):
      new_layer_group = new_model.encoder_network.transformer_layers
    else:
      raise ValueError('Unrecognized encoder network: {}'.format(
          new_model.encoder_network))
    for new_layer_idx in range(len(new_layer_group)):
      old_layer_idx = new_layer_idx % len(old_layer_group)
      new_layer_group[new_layer_idx].set_weights(
          old_layer_group[old_layer_idx].get_weights())
      if old_layer_idx != new_layer_idx:
        if hasattr(new_layer_group[new_layer_idx], 'reset_rezero'):
          # Reset ReZero's alpha to 0.
          new_layer_group[new_layer_idx].reset_rezero()

    # Copy weights of the final layer norm (if needed).
    # pylint: disable=protected-access
    if hasattr(new_model.encoder_network, '_output_layer_norm') and hasattr(
        old_model.encoder_network, '_output_layer_norm'):
      new_model.encoder_network._output_layer_norm.set_weights(
          old_model.encoder_network._output_layer_norm.get_weights())
    # pylint: enable=protected-access

    # Copy weights of the pooler layer.
    new_model.encoder_network.pooler_layer.set_weights(
        old_model.encoder_network.pooler_layer.get_weights())

    # Copy weights of the classification head.
    for idx in range(len(new_model.classification_heads)):
      new_model.classification_heads[idx].set_weights(
          old_model.classification_heads[idx].get_weights())

    # Copy weights of the masked_lm layer.
    new_model.masked_lm.set_weights(old_model.masked_lm.get_weights())
