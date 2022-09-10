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

"""SimCLR configurations."""
import dataclasses
import os
from typing import List, Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.projects.simclr.modeling import simclr_model
from official.vision.configs import backbones
from official.vision.configs import common


@dataclasses.dataclass
class Decoder(hyperparams.Config):
  decode_label: bool = True


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Parser config."""
  aug_rand_crop: bool = True
  aug_rand_hflip: bool = True
  aug_color_distort: bool = True
  aug_color_jitter_strength: float = 1.0
  aug_color_jitter_impl: str = 'simclrv2'  # 'simclrv1' or 'simclrv2'
  aug_rand_blur: bool = True
  parse_label: bool = True
  test_crop: bool = True
  mode: str = simclr_model.PRETRAIN


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Training data config."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  # simclr specific configs
  parser: Parser = Parser()
  decoder: Decoder = Decoder()
  # Useful when doing a sanity check that we absolutely use no labels while
  # pretrain by setting labels to zeros (default = False, keep original labels)
  input_set_label_to_zero: bool = False


@dataclasses.dataclass
class ProjectionHead(hyperparams.Config):
  proj_output_dim: int = 128
  num_proj_layers: int = 3
  ft_proj_idx: int = 1  # layer of the projection head to use for fine-tuning.


@dataclasses.dataclass
class SupervisedHead(hyperparams.Config):
  num_classes: int = 1001
  zero_init: bool = False


@dataclasses.dataclass
class ContrastiveLoss(hyperparams.Config):
  projection_norm: bool = True
  temperature: float = 0.1
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class ClassificationLosses(hyperparams.Config):
  label_smoothing: float = 0.0
  one_hot: bool = True
  l2_weight_decay: float = 0.0


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  top_k: int = 5
  one_hot: bool = True


@dataclasses.dataclass
class SimCLRModel(hyperparams.Config):
  """SimCLR model config."""
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  projection_head: ProjectionHead = ProjectionHead(
      proj_output_dim=128, num_proj_layers=3, ft_proj_idx=1)
  supervised_head: SupervisedHead = SupervisedHead(num_classes=1001)
  norm_activation: common.NormActivation = common.NormActivation(
      norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)
  mode: str = simclr_model.PRETRAIN
  backbone_trainable: bool = True


@dataclasses.dataclass
class SimCLRPretrainTask(cfg.TaskConfig):
  """SimCLR pretraining task config."""
  model: SimCLRModel = SimCLRModel(mode=simclr_model.PRETRAIN)
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.PRETRAIN), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.PRETRAIN), is_training=False)
  loss: ContrastiveLoss = ContrastiveLoss()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  # all or backbone
  init_checkpoint_modules: str = 'all'


@dataclasses.dataclass
class SimCLRFinetuneTask(cfg.TaskConfig):
  """SimCLR fine tune task config."""
  model: SimCLRModel = SimCLRModel(
      mode=simclr_model.FINETUNE,
      supervised_head=SupervisedHead(num_classes=1001, zero_init=True))
  train_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.FINETUNE), is_training=True)
  validation_data: DataConfig = DataConfig(
      parser=Parser(mode=simclr_model.FINETUNE), is_training=False)
  loss: ClassificationLosses = ClassificationLosses()
  evaluation: Evaluation = Evaluation()
  init_checkpoint: Optional[str] = None
  # all, backbone_projection or backbone
  init_checkpoint_modules: str = 'backbone_projection'


@exp_factory.register_config_factory('simclr_pretraining')
def simclr_pretraining() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRPretrainTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


@exp_factory.register_config_factory('simclr_finetuning')
def simclr_finetuning() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=SimCLRFinetuneTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


@exp_factory.register_config_factory('simclr_pretraining_imagenet')
def simclr_pretraining_imagenet() -> cfg.ExperimentConfig:
  """Image classification general."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  return cfg.ExperimentConfig(
      task=SimCLRPretrainTask(
          model=SimCLRModel(
              mode=simclr_model.PRETRAIN,
              backbone_trainable=True,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              projection_head=ProjectionHead(
                  proj_output_dim=128, num_proj_layers=3, ft_proj_idx=1),
              supervised_head=SupervisedHead(num_classes=1001),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=True)),
          loss=ContrastiveLoss(),
          evaluation=Evaluation(),
          train_data=DataConfig(
              parser=Parser(mode=simclr_model.PRETRAIN),
              decoder=Decoder(decode_label=True),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              parser=Parser(mode=simclr_model.PRETRAIN),
              decoder=Decoder(decode_label=True),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size),
      ),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=500 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'lars',
                  'lars': {
                      'momentum':
                          0.9,
                      'weight_decay_rate':
                          0.000001,
                      'exclude_from_weight_decay': [
                          'batch_normalization', 'bias'
                      ]
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      # 0.2 * BatchSize / 256
                      'initial_learning_rate': 0.2 * train_batch_size / 256,
                      # train_steps - warmup_steps
                      'decay_steps': 475 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      # 5% of total epochs
                      'warmup_steps': 25 * steps_per_epoch
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])


@exp_factory.register_config_factory('simclr_finetuning_imagenet')
def simclr_finetuning_imagenet() -> cfg.ExperimentConfig:
  """Image classification general."""
  train_batch_size = 1024
  eval_batch_size = 1024
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  pretrain_model_base = ''
  return cfg.ExperimentConfig(
      task=SimCLRFinetuneTask(
          model=SimCLRModel(
              mode=simclr_model.FINETUNE,
              backbone_trainable=True,
              input_size=[224, 224, 3],
              backbone=backbones.Backbone(
                  type='resnet', resnet=backbones.ResNet(model_id=50)),
              projection_head=ProjectionHead(
                  proj_output_dim=128, num_proj_layers=3, ft_proj_idx=1),
              supervised_head=SupervisedHead(num_classes=1001, zero_init=True),
              norm_activation=common.NormActivation(
                  norm_momentum=0.9, norm_epsilon=1e-5, use_sync_bn=False)),
          loss=ClassificationLosses(),
          evaluation=Evaluation(),
          train_data=DataConfig(
              parser=Parser(mode=simclr_model.FINETUNE),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size),
          validation_data=DataConfig(
              parser=Parser(mode=simclr_model.FINETUNE),
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size),
          init_checkpoint=pretrain_model_base,
          # all, backbone_projection or backbone
          init_checkpoint_modules='backbone_projection'),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=60 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'lars',
                  'lars': {
                      'momentum':
                          0.9,
                      'weight_decay_rate':
                          0.0,
                      'exclude_from_weight_decay': [
                          'batch_normalization', 'bias'
                      ]
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      # 0.01 × BatchSize / 512
                      'initial_learning_rate': 0.01 * train_batch_size / 512,
                      'decay_steps': 60 * steps_per_epoch
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
