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

"""Spatiotemporal action localization configuration definition."""

import dataclasses
from typing import Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.projects.mae import optimization
from official.projects.videoglue.configs import backbones_3d
from official.projects.videoglue.configs import dataset
from official.projects.videoglue.configs import head as head_cfg
from official.vision.configs import common
from official.vision.configs import video_classification

Losses = video_classification.Losses


@dataclasses.dataclass
class DataConfig(dataset.DataConfig):
  """The dataset config."""
  data_augmentation: dataset.DataAugmentation = dataclasses.field(
      default_factory=lambda: dataset.DataAugmentation(  # pylint: disable=g-long-lambda
          type='ava', ava=dataset.AVA()
      )
  )
  is_training: bool = True
  drop_remainder: bool = True
  num_instances: int = 32
  num_classes: int = 80
  one_hot_label: bool = True
  merge_multi_labels: bool = True
  import_detected_bboxes: bool = False
  color_augmentation: bool = True


@dataclasses.dataclass
class VideoActionTransformerModel(hyperparams.Config):
  """The model config."""
  model_type: str = 'video_action_transformer_model'
  backbone: backbones_3d.Backbone3D = dataclasses.field(
      default_factory=lambda: backbones_3d.Backbone3D(  # pylint: disable=g-long-lambda
          type='vit_3d',
          vit_3d=backbones_3d.VisionTransformer3D(pooler='none')))
  endpoint_name: str = 'encoded_tokens'
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=lambda: common.NormActivation(  # pylint: disable=g-long-lambda
          use_sync_bn=True, norm_momentum=0.9, norm_epsilon=1e-5
      )
  )
  head: head_cfg.ActionTransformer = dataclasses.field(
      default_factory=lambda: head_cfg.ActionTransformer(  # pylint: disable=g-long-lambda
          use_sync_bn=True,
          num_hidden_layers=1,
          num_hidden_channels=1024,
          crop_size=7,
      )
  )


@dataclasses.dataclass
class SpatiotemporalActionLocalizationTask(
    video_classification.VideoClassificationTask):
  """Task for video action localization."""
  model: VideoActionTransformerModel = dataclasses.field(
      default_factory=VideoActionTransformerModel
  )
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(  # pylint: disable=g-long-lambda
          data_augmentation=dataset.DataAugmentation(type='vgg'),
          is_training=True,
          drop_remainder=True,
      )
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(  # pylint: disable=g-long-lambda
          is_training=False, drop_remainder=False
      )
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone


# NOTE: This utility function includes different name conventions of the same
# layer from different ViT implementation. They would be safely ignored as long
# as no substring collision happens. However it is still error-prone. Use with
# caution. See how it is in:
# tensorflow_models/official/projects/mae/optimization.py;l=40
def _get_vit_layers(num_tx_layers: int = 12):
  """Gets ViT layers substring and index."""
  layers_substr = [
      # rgb projection layer supports VMAE/INTERNVIDEO/FLVID.
      'conv3d/kernel',
      'conv3d/bias',

      # postional embedding for VMAE.
      'add_separable_position_embs/pos_embedding_time',
      'add_separable_position_embs/pos_embedding_space',

      # rgb projection layer supports IMP.
      'rgb_to_embedding',
      # positional embedding for IMP.
      'rgb_pos_encoding',
      # pre-projection for IMP.
      'pre_projection/vision_dense',

      # rgb projection layer supports COCA.
      'input_projection/kernel',
      'input_projection/bias',
      # rgb projection layer supports FLAVA.
      'conv2d/kernel',
      'conv2d/bias',

      # positional embedding for COCA/FLAVA.
      'encoder/posembed_input/pos_embedding',

      # common encoder final layer norm.
      'encoder/layer_normalization',

      # post-projection for IMP.
      'post_projection/vision_dense',
  ]
  layers_idx = [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      num_tx_layers,
      num_tx_layers,
  ]
  if len(layers_idx) != len(layers_substr):
    raise ValueError('layers_idx and layers_substr should have same length.')

  for idx in range(num_tx_layers):
    if idx == 0:
      var_substr = 'encoder/transformer_encoder_block/'
    else:
      var_substr = f'encoder/transformer_encoder_block_{idx}/'
    layers_substr.append(var_substr)
    layers_idx.append(idx + 1)
  return layers_idx, layers_substr


def _get_clip_layers(num_tx_layers: int = 12):
  """Gets CLIP layers substring and index."""
  layers_substr = [
      # rgb projection layer
      'conv1/kernel',

      # postional embedding.
      'clip/visual/positional_embedding',

      # encoder pre layer norm.
      'ln_pre',

      # class embedding.
      'clip/visual/class_embedding',

      # post layer norm.
      'ln_post',

      # post-projection.
      'proj/kernel',
  ]
  layers_idx = [
      0,
      0,
      0,
      0,
      num_tx_layers,
      num_tx_layers,
  ]
  if len(layers_idx) != len(layers_substr):
    raise ValueError('layers_idx and layers_substr should have same length.')

  for idx in range(num_tx_layers):
    var_substr = f'transformer/resblocks.{idx}/'
    layers_substr.append(var_substr)
    layers_idx.append(idx + 1)
  return layers_idx, layers_substr


@exp_factory.register_config_factory('spatiotemporal_action_localization')
def spatiotemporal_action_localization() -> cfg.ExperimentConfig:
  """Spatio-temporal action localization."""
  task = SpatiotemporalActionLocalizationTask()
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=task,
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.train_data.num_classes == task.validation_data.num_classes',
      ])
  config = video_classification.add_trainer(
      config, train_batch_size=1024, eval_batch_size=64)
  return config


@exp_factory.register_config_factory('spatiotemporal_action_localization_vit12')
def spatiotemporal_action_localization_vit12() -> cfg.ExperimentConfig:
  """Spatio-temporal action localization for ViT-B with layer decay."""
  config = spatiotemporal_action_localization()
  layers_idx, vars_substr = _get_vit_layers(num_tx_layers=12)
  optimizer_config = optimization.OptimizerConfig({
      'type': 'vit_adamw',
      'vit_adamw': {
          'weight_decay_rate': 0.05,
          # Avoid AdamW legacy behavior.
          'gradient_clip_norm': 0.0,
          'beta_2': 0.999,
          'layer_decay': 0.75,
          'vars_substr': vars_substr,
          'layers_idx': layers_idx,
          'exclude_from_weight_decay': ['cls'],
      },
  })
  config.trainer.optimizer_config.optimizer = optimizer_config
  return config


@exp_factory.register_config_factory(
    'spatiotemporal_action_localization_clip12')
def spatiotemporal_action_localization_clip12() -> cfg.ExperimentConfig:
  """Spatio-temporal action localization for CLIP-B with layer decay."""
  config = spatiotemporal_action_localization()
  layers_idx, vars_substr = _get_clip_layers(num_tx_layers=12)
  optimizer_config = optimization.OptimizerConfig({
      'type': 'vit_adamw',
      'vit_adamw': {
          'weight_decay_rate': 0.05,
          # Avoid AdamW legacy behavior.
          'gradient_clip_norm': 0.0,
          'beta_2': 0.999,
          'layer_decay': 0.75,
          'vars_substr': vars_substr,
          'layers_idx': layers_idx,
          'exclude_from_weight_decay': ['cls'],
      },
  })
  config.trainer.optimizer_config.optimizer = optimizer_config
  return config


@exp_factory.register_config_factory('spatiotemporal_action_localization_vit16')
def spatiotemporal_action_localization_vit16() -> cfg.ExperimentConfig:
  """Spatio-temporal action localization for ViT-L/H/G with layer decay."""
  config = spatiotemporal_action_localization()
  # ViT-L/H/G have 16 layers transformer encoder block.
  layers_idx, vars_substr = _get_vit_layers(num_tx_layers=16)
  optimizer_config = optimization.OptimizerConfig({
      'type': 'vit_adamw',
      'vit_adamw': {
          'weight_decay_rate': 0.05,
          # Avoid AdamW legacy behavior.
          'gradient_clip_norm': 0.0,
          'beta_2': 0.999,
          'layer_decay': 0.75,
          'vars_substr': vars_substr,
          'layers_idx': layers_idx,
          'exclude_from_weight_decay': ['cls'],
      },
  })
  config.trainer.optimizer_config.optimizer = optimizer_config
  return config
