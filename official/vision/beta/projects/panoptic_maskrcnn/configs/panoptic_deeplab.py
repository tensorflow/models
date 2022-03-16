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

"""Panoptic Deeplab configuration definition."""

import dataclasses
from typing import List, Optional, Tuple, Union

from official.core import config_definitions as cfg
from official.modeling import hyperparams
from official.vision.beta.configs import common
from official.vision.beta.configs import backbones
from official.vision.beta.configs import decoders


_COCO_INPUT_PATH_BASE = 'coco/tfrecords'
_COCO_TRAIN_EXAMPLES = 118287
_COCO_VAL_EXAMPLES = 5000


@dataclasses.dataclass
class Parser(hyperparams.Config):
  ignore_label: int = 0
  # If resize_eval_groundtruth is set to False, original image sizes are used
  # for eval. In that case, groundtruth_padded_size has to be specified too to
  # allow for batching the variable input sizes of images.
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_hflip: bool = True
  sigma: float = 8.0
  dtype = 'float32'

@dataclasses.dataclass
class DataDecoder(common.DataDecoder):
  """Data decoder config."""
  simple_decoder: common.TfExampleDecoder = common.TfExampleDecoder()

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  file_type: str = 'tfrecord'

@dataclasses.dataclass
class PanopticDeeplabHead(hyperparams.Config):
  """Panoptic Deeplab head config."""
  level: int = 3
  num_convs: int = 2
  num_filters: int = 256
  kernel_size: int = 5
  use_depthwise_convolution: bool = False
  upsample_factor: int = 1
  low_level: Union[List[int], Tuple[int]] = (3, 2)
  low_level_num_filters: Union[List[int], Tuple[int]] = (64, 32)

@dataclasses.dataclass
class SemanticHead(PanopticDeeplabHead):
  """Semantic head config."""
  prediction_kernel_size: int = 1

@dataclasses.dataclass
class InstanceHead(PanopticDeeplabHead):
  """Instance head config."""
  prediction_kernel_size: int = 1

@dataclasses.dataclass
class PanopticDeeplabPostProcessor(hyperparams.Config):
  """Panoptic Deeplab PostProcessing config."""
  output_size: List[int] = dataclasses.field(
      default_factory=list)
  center_score_threshold: float = 0.1
  thing_class_ids: List[int] = dataclasses.field(default_factory=list)
  label_divisor: int = 256 * 256 * 256
  stuff_area_limit: int = 4096
  ignore_label: int = 0
  nms_kernel: int = 41
  keep_k_centers: int = 400
  rescale_predictions: bool = True

@dataclasses.dataclass
class PanopticDeeplab(hyperparams.Config):
  """Panoptic Deeplab model config."""
  num_classes: int = 2
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 6
  norm_activation: common.NormActivation = common.NormActivation()
  backbone: backbones.Backbone = backbones.Backbone(
      type='resnet', resnet=backbones.ResNet())
  decoder: decoders.Decoder = decoders.Decoder(type='aspp')
  semantic_head: SemanticHead = SemanticHead()
  instance_head: InstanceHead = InstanceHead()
  shared_decoder: bool = False
  generate_panoptic_masks: bool = True
  post_processor: PanopticDeeplabPostProcessor = PanopticDeeplabPostProcessor()

@dataclasses.dataclass
class Losses(hyperparams.Config):
  label_smoothing: float = 0.0
  ignore_label: int = 0
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 1e-4
  use_groundtruth_dimension: bool = True
  top_k_percent_pixels: float = 0.15
  segmentation_loss_weight: float = 1.0
  center_heatmap_loss_weight: float = 200
  center_offset_loss_weight: float = 0.01

@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  """ Evaluation config """
  ignored_label: int = 0
  max_instances_per_category: int = 256
  offset: int = 256 * 256 * 256
  is_thing: List[float] = dataclasses.field(
      default_factory=list)
  rescale_predictions: bool = True
  report_per_class_pq: bool = False

  report_per_class_iou: bool = False
  report_train_mean_iou: bool = True  # Turning this off can speed up training.

@dataclasses.dataclass
class PanopticDeeplabTask(cfg.TaskConfig):
  model: PanopticDeeplab = PanopticDeeplab()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(
      is_training=False,
      drop_remainder=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[str, List[str]] = 'all'  # all, backbone, and/or decoder
  annotation_file: Optional[str] = None
  evaluation: Evaluation = Evaluation()
