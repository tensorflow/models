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

"""Pointpillars experiment configuration definition."""
import dataclasses
from typing import List, Optional, Tuple, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common


@dataclasses.dataclass
class ImageConfig(hyperparams.Config):
  """Bird-eye-view pseudo image config."""
  # The range should be large enough to cover a 64-channels Lidar points.
  # The default values are chosen empirically.
  x_range: Tuple[float, float] = (-76.8, 76.8)
  y_range: Tuple[float, float] = (-76.8, 76.8)
  z_range: Tuple[float, float] = (-3.0, 3.0)
  resolution: float = 0.3
  height: int = dataclasses.field(init=False)
  width: int = dataclasses.field(init=False)

  # Image height and width should be auto computed.
  def __post_init__(self, height: int, width: int):
    self.height = int((-self.x_range[0] + self.x_range[1]) / self.resolution)
    self.width = int((-self.y_range[0] + self.y_range[1]) / self.resolution)


@dataclasses.dataclass
class PillarsConfig(hyperparams.Config):
  """Pillars config."""
  num_pillars: int = 24000
  num_points_per_pillar: int = 100
  num_features_per_point: int = 10


@dataclasses.dataclass
class DataDecoder(hyperparams.Config):
  """Data decoder config."""


@dataclasses.dataclass
class DataParser(hyperparams.Config):
  """Data parser config."""


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'float32'
  decoder: DataDecoder = dataclasses.field(default_factory=DataDecoder)
  parser: DataParser = dataclasses.field(default_factory=DataParser)
  shuffle_buffer_size: int = 256
  prefetch_buffer_size: int = 256
  file_type: str = 'tfrecord_compressed'


@dataclasses.dataclass
class Anchor(hyperparams.Config):
  length: float = 1.0
  width: float = 1.0


@dataclasses.dataclass
class AnchorLabeler(hyperparams.Config):
  """Data parser config."""
  match_threshold: float = 0.5
  unmatched_threshold: float = 0.5


@dataclasses.dataclass
class Featurizer(hyperparams.Config):
  num_blocks: int = 1
  num_channels: int = 64


@dataclasses.dataclass
class Backbone(hyperparams.Config):
  min_level: int = 1
  max_level: int = 3
  num_convs: int = 6


@dataclasses.dataclass
class Decoder(hyperparams.Config):
  """Feature decoder."""
  # No fields yet, just a placeholder.


@dataclasses.dataclass
class AttributeHead(hyperparams.Config):
  name: str = ''
  type: str = 'regression'
  size: int = 1


def _default_heads():
  return [
      AttributeHead(name='heading', type='regression', size=1),
      AttributeHead(name='height', type='regression', size=1),
      AttributeHead(name='z', type='regression', size=1)
  ]


@dataclasses.dataclass
class SSDHead(hyperparams.Config):
  attribute_heads: List[AttributeHead] = dataclasses.field(
      default_factory=_default_heads)


@dataclasses.dataclass
class DetectionGenerator(hyperparams.Config):
  """Generator."""
  apply_nms: bool = True
  pre_nms_top_k: int = 5000
  pre_nms_score_threshold: float = 0.05
  nms_iou_threshold: float = 0.5
  max_num_detections: int = 100
  nms_version: str = 'v1'  # `v2`, `v1`, `batched`
  use_cpu_nms: bool = False


@dataclasses.dataclass
class PointPillarsModel(hyperparams.Config):
  """The model config. Used by build_example_model function."""
  classes: str = 'all'
  num_classes: int = 4
  image: ImageConfig = dataclasses.field(default_factory=ImageConfig)
  pillars: PillarsConfig = dataclasses.field(default_factory=PillarsConfig)
  anchors: List[Anchor] = dataclasses.field(default_factory=list)
  anchor_labeler: AnchorLabeler = dataclasses.field(
      default_factory=AnchorLabeler
  )

  min_level: int = 1
  max_level: int = 3
  featurizer: Featurizer = dataclasses.field(default_factory=Featurizer)
  backbone: Backbone = dataclasses.field(default_factory=Backbone)
  decoder: Decoder = dataclasses.field(default_factory=Decoder)
  head: SSDHead = dataclasses.field(default_factory=SSDHead)
  detection_generator: DetectionGenerator = dataclasses.field(
      default_factory=DetectionGenerator
  )
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=common.NormActivation
  )


@dataclasses.dataclass
class Losses(hyperparams.Config):
  loss_weight: float = 1.0
  box_loss_weight: int = 100
  attribute_loss_weight: int = 10
  focal_loss_alpha: float = 0.25
  focal_loss_gamma: float = 1.5
  huber_loss_delta: float = 0.1
  l2_weight_decay: float = 0


@dataclasses.dataclass
class PointPillarsTask(cfg.TaskConfig):
  """The task config."""
  model: PointPillarsModel = dataclasses.field(
      default_factory=PointPillarsModel
  )
  use_raw_data: bool = False
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=False)
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = 'all'
  use_wod_metrics: bool = True


@exp_factory.register_config_factory('pointpillars_baseline')
def pointpillars_baseline() -> cfg.ExperimentConfig:
  """PointPillars baseline config."""
  return cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=PointPillarsTask(
          model=PointPillarsModel(
              classes='vehicle',
              num_classes=2,
              min_level=1,
              max_level=1,
              anchors=[Anchor(length=1.0, width=1.0)],
              featurizer=Featurizer(),
              backbone=Backbone(),
              decoder=Decoder(),
              head=SSDHead()
          ),
          train_data=DataConfig(is_training=True),
          validation_data=DataConfig(is_training=False),
          losses=Losses()
      ),
      trainer=cfg.TrainerConfig(
          train_steps=100,
          validation_steps=100,
          validation_interval=10,
          steps_per_loop=10,
          summary_interval=10,
          checkpoint_interval=10,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'decay_steps': 100,
                      'initial_learning_rate': 0.16,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 10,
                      'warmup_learning_rate': 0.016
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])
