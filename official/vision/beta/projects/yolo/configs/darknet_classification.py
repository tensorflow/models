import os
from typing import List, Optional
import dataclasses
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.beta.projects.yolo.configs import backbones
from official.vision.beta.configs import common
from official.vision.beta.configs import image_classification as imc


@dataclasses.dataclass
class ImageClassificationModel(hyperparams.Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  backbone: backbones.Backbone = backbones.Backbone(
      type='darknet', resnet=backbones.DarkNet())
  dropout_rate: float = 0.0
  norm_activation: common.NormActivation = common.NormActivation()
  # Adds a BatchNormalization layer pre-GlobalAveragePooling in classification
  add_head_batch_norm: bool = False


@dataclasses.dataclass
class Losses(hyperparams.Config):
  one_hot: bool = True
  label_smoothing: float = 0.0
  l2_weight_decay: float = 0.0

@dataclasses.dataclass
class ImageClassificationTask(cfg.TaskConfig):
  """The model config."""
  model: ImageClassificationModel = ImageClassificationModel()
  train_data: imc.DataConfig = imc.DataConfig(is_training=True)
  validation_data: imc.DataConfig = imc.DataConfig(is_training=False)
  losses: Losses = Losses()
  gradient_clip_norm: float = 0.0
  logging_dir:str = None


@exp_factory.register_config_factory('darknet_classification')
def image_classification() -> cfg.ExperimentConfig:
  """Image classification general."""
  return cfg.ExperimentConfig(
      task=ImageClassificationTask(),
      trainer=cfg.TrainerConfig(),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
