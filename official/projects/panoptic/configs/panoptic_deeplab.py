# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
import math
import os
from typing import List, Optional, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones


_COCO_INPUT_PATH_BASE = 'coco/tfrecords'
_COCO_TRAIN_EXAMPLES = 118287
_COCO_VAL_EXAMPLES = 5000


@dataclasses.dataclass
class Parser(hyperparams.Config):
  """Panoptic deeplab parser."""
  ignore_label: int = 0
  # If resize_eval_groundtruth is set to False, original image sizes are used
  # for eval. In that case, groundtruth_padded_size has to be specified too to
  # allow for batching the variable input sizes of images.
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_hflip: bool = True
  aug_type: common.Augmentation = dataclasses.field(
      default_factory=common.Augmentation
  )
  sigma: float = 8.0
  small_instance_area_threshold: int = 4096
  small_instance_weight: float = 3.0
  dtype = 'float32'


@dataclasses.dataclass
class TfExampleDecoder(common.TfExampleDecoder):
  """A simple TF Example decoder config."""
  panoptic_category_mask_key: str = 'image/panoptic/category_mask'
  panoptic_instance_mask_key: str = 'image/panoptic/instance_mask'


@dataclasses.dataclass
class DataDecoder(common.DataDecoder):
  """Data decoder config."""
  simple_decoder: TfExampleDecoder = dataclasses.field(
      default_factory=TfExampleDecoder
  )


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  decoder: DataDecoder = dataclasses.field(default_factory=DataDecoder)
  parser: Parser = dataclasses.field(default_factory=Parser)
  input_path: str = ''
  drop_remainder: bool = True
  file_type: str = 'tfrecord'
  is_training: bool = True
  global_batch_size: int = 1


@dataclasses.dataclass
class PanopticDeeplabHead(hyperparams.Config):
  """Panoptic Deeplab head config."""
  level: int = 3
  num_convs: int = 2
  num_filters: int = 256
  kernel_size: int = 5
  use_depthwise_convolution: bool = False
  upsample_factor: int = 1
  low_level: List[int] = dataclasses.field(default_factory=lambda: [3, 2])
  low_level_num_filters: List[int] = dataclasses.field(
      default_factory=lambda: [64, 32])
  fusion_num_output_filters: int = 256


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
  nms_kernel: int = 7
  keep_k_centers: int = 200
  rescale_predictions: bool = True


@dataclasses.dataclass
class PanopticDeeplab(hyperparams.Config):
  """Panoptic Deeplab model config."""
  num_classes: int = 2
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 3
  max_level: int = 6
  norm_activation: common.NormActivation = dataclasses.field(
      default_factory=common.NormActivation
  )
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(
          type='resnet', resnet=backbones.ResNet()
      )
  )
  decoder: decoders.Decoder = dataclasses.field(
      default_factory=lambda: decoders.Decoder(
          type='aspp', aspp=decoders.ASPP(level=3)
      )
  )
  semantic_head: SemanticHead = dataclasses.field(default_factory=SemanticHead)
  instance_head: InstanceHead = dataclasses.field(default_factory=InstanceHead)
  shared_decoder: bool = False
  generate_panoptic_masks: bool = True
  post_processor: PanopticDeeplabPostProcessor = dataclasses.field(
      default_factory=PanopticDeeplabPostProcessor
  )


@dataclasses.dataclass
class Losses(hyperparams.Config):
  label_smoothing: float = 0.0
  ignore_label: int = 0
  class_weights: List[float] = dataclasses.field(default_factory=list)
  l2_weight_decay: float = 1e-4
  top_k_percent_pixels: float = 0.15
  segmentation_loss_weight: float = 1.0
  center_heatmap_loss_weight: float = 200
  center_offset_loss_weight: float = 0.01


@dataclasses.dataclass
class Evaluation(hyperparams.Config):
  """Evaluation config."""
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
  """Panoptic deeplab task config."""
  model: PanopticDeeplab = dataclasses.field(default_factory=PanopticDeeplab)
  train_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(is_training=True)
  )
  validation_data: DataConfig = dataclasses.field(
      default_factory=lambda: DataConfig(  # pylint: disable=g-long-lambda
          is_training=False, drop_remainder=False
      )
  )
  losses: Losses = dataclasses.field(default_factory=Losses)
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder
  evaluation: Evaluation = dataclasses.field(default_factory=Evaluation)


@exp_factory.register_config_factory('panoptic_deeplab_resnet_coco')
def panoptic_deeplab_resnet_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with Panoptic Deeplab."""
  train_steps = 200000
  train_batch_size = 64
  eval_batch_size = 1
  steps_per_epoch = _COCO_TRAIN_EXAMPLES // train_batch_size
  validation_steps = _COCO_VAL_EXAMPLES // eval_batch_size

  num_panoptic_categories = 201
  num_thing_categories = 91
  ignore_label = 0

  is_thing = [False]
  for idx in range(1, num_panoptic_categories):
    is_thing.append(True if idx <= num_thing_categories else False)

  input_size = [640, 640, 3]
  output_stride = 16
  aspp_dilation_rates = [6, 12, 18]
  multigrid = [1, 2, 4]
  stem_type = 'v1'
  level = int(math.log2(output_stride))

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='bfloat16', enable_xla=True),
      task=PanopticDeeplabTask(
          init_checkpoint='gs://tf_model_garden/vision/panoptic/panoptic_deeplab/imagenet/resnet50_v1/ckpt-436800',  # pylint: disable=line-too-long
          init_checkpoint_modules=['backbone'],
          model=PanopticDeeplab(
              num_classes=num_panoptic_categories,
              input_size=input_size,
              backbone=backbones.Backbone(
                  type='dilated_resnet', dilated_resnet=backbones.DilatedResNet(
                      model_id=50,
                      stem_type=stem_type,
                      output_stride=output_stride,
                      multigrid=multigrid,
                      se_ratio=0.25,
                      last_stage_repeats=1,
                      stochastic_depth_drop_rate=0.2)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level,
                      num_filters=256,
                      pool_kernel_size=input_size[:2],
                      dilation_rates=aspp_dilation_rates,
                      use_depthwise_convolution=True,
                      dropout_rate=0.1)),
              semantic_head=SemanticHead(
                  level=level,
                  num_convs=1,
                  num_filters=256,
                  kernel_size=5,
                  use_depthwise_convolution=True,
                  upsample_factor=1,
                  low_level=[3, 2],
                  low_level_num_filters=[64, 32],
                  fusion_num_output_filters=256,
                  prediction_kernel_size=1),
              instance_head=InstanceHead(
                  level=level,
                  num_convs=1,
                  num_filters=32,
                  kernel_size=5,
                  use_depthwise_convolution=True,
                  upsample_factor=1,
                  low_level=[3, 2],
                  low_level_num_filters=[32, 16],
                  fusion_num_output_filters=128,
                  prediction_kernel_size=1),
              shared_decoder=False,
              generate_panoptic_masks=True,
              post_processor=PanopticDeeplabPostProcessor(
                  output_size=input_size[:2],
                  center_score_threshold=0.1,
                  thing_class_ids=list(range(1, num_thing_categories)),
                  label_divisor=256,
                  stuff_area_limit=4096,
                  ignore_label=ignore_label,
                  nms_kernel=41,
                  keep_k_centers=200,
                  rescale_predictions=True)),
          losses=Losses(
              label_smoothing=0.0,
              ignore_label=ignore_label,
              l2_weight_decay=0.0,
              top_k_percent_pixels=0.2,
              segmentation_loss_weight=1.0,
              center_heatmap_loss_weight=200,
              center_offset_loss_weight=0.01),
          train_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_scale_min=0.5,
                  aug_scale_max=1.5,
                  aug_rand_hflip=True,
                  aug_type=common.Augmentation(
                      type='autoaug',
                      autoaug=common.AutoAugment(
                          augmentation_name='panoptic_deeplab_policy')),
                  sigma=8.0,
                  small_instance_area_threshold=4096,
                  small_instance_weight=3.0)),
          validation_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              parser=Parser(
                  resize_eval_groundtruth=False,
                  groundtruth_padded_size=[640, 640],
                  aug_scale_min=1.0,
                  aug_scale_max=1.0,
                  aug_rand_hflip=False,
                  aug_type=None,
                  sigma=8.0,
                  small_instance_area_threshold=4096,
                  small_instance_weight=3.0),
              drop_remainder=False),
          evaluation=Evaluation(
              ignored_label=ignore_label,
              max_instances_per_category=256,
              offset=256*256*256,
              is_thing=is_thing,
              rescale_predictions=True,
              report_per_class_pq=False,
              report_per_class_iou=False,
              report_train_mean_iou=False)),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=validation_steps,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.0005,
                      'decay_steps': train_steps,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('panoptic_deeplab_mobilenetv3_large_coco')
def panoptic_deeplab_mobilenetv3_large_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with Panoptic Deeplab."""
  train_steps = 200000
  train_batch_size = 64
  eval_batch_size = 1
  steps_per_epoch = _COCO_TRAIN_EXAMPLES // train_batch_size
  validation_steps = _COCO_VAL_EXAMPLES // eval_batch_size

  num_panoptic_categories = 201
  num_thing_categories = 91
  ignore_label = 0

  is_thing = [False]
  for idx in range(1, num_panoptic_categories):
    is_thing.append(True if idx <= num_thing_categories else False)

  input_size = [640, 640, 3]
  output_stride = 16
  aspp_dilation_rates = [6, 12, 18]
  level = int(math.log2(output_stride))

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='float32', enable_xla=True),
      task=PanopticDeeplabTask(
          init_checkpoint='gs://tf_model_garden/vision/panoptic/panoptic_deeplab/imagenet/mobilenetv3_large/ckpt-156000',
          init_checkpoint_modules=['backbone'],
          model=PanopticDeeplab(
              num_classes=num_panoptic_categories,
              input_size=input_size,
              backbone=backbones.Backbone(
                  type='mobilenet', mobilenet=backbones.MobileNet(
                      model_id='MobileNetV3Large',
                      filter_size_scale=1.0,
                      stochastic_depth_drop_rate=0.0,
                      output_stride=output_stride)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level,
                      num_filters=256,
                      pool_kernel_size=input_size[:2],
                      dilation_rates=aspp_dilation_rates,
                      use_depthwise_convolution=True,
                      dropout_rate=0.1)),
              semantic_head=SemanticHead(
                  level=level,
                  num_convs=1,
                  num_filters=256,
                  kernel_size=5,
                  use_depthwise_convolution=True,
                  upsample_factor=1,
                  low_level=[3, 2],
                  low_level_num_filters=[64, 32],
                  fusion_num_output_filters=256,
                  prediction_kernel_size=1),
              instance_head=InstanceHead(
                  level=level,
                  num_convs=1,
                  num_filters=32,
                  kernel_size=5,
                  use_depthwise_convolution=True,
                  upsample_factor=1,
                  low_level=[3, 2],
                  low_level_num_filters=[32, 16],
                  fusion_num_output_filters=128,
                  prediction_kernel_size=1),
              shared_decoder=False,
              generate_panoptic_masks=True,
              post_processor=PanopticDeeplabPostProcessor(
                  output_size=input_size[:2],
                  center_score_threshold=0.1,
                  thing_class_ids=list(range(1, num_thing_categories)),
                  label_divisor=256,
                  stuff_area_limit=4096,
                  ignore_label=ignore_label,
                  nms_kernel=41,
                  keep_k_centers=200,
                  rescale_predictions=True)),
          losses=Losses(
              label_smoothing=0.0,
              ignore_label=ignore_label,
              l2_weight_decay=0.0,
              top_k_percent_pixels=0.2,
              segmentation_loss_weight=1.0,
              center_heatmap_loss_weight=200,
              center_offset_loss_weight=0.01),
          train_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_scale_min=0.5,
                  aug_scale_max=2.0,
                  aug_rand_hflip=True,
                  aug_type=common.Augmentation(
                      type='autoaug',
                      autoaug=common.AutoAugment(
                          augmentation_name='panoptic_deeplab_policy')),
                  sigma=8.0,
                  small_instance_area_threshold=4096,
                  small_instance_weight=3.0)),
          validation_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              parser=Parser(
                  resize_eval_groundtruth=False,
                  groundtruth_padded_size=[640, 640],
                  aug_scale_min=1.0,
                  aug_scale_max=1.0,
                  aug_rand_hflip=False,
                  aug_type=None,
                  sigma=8.0,
                  small_instance_area_threshold=4096,
                  small_instance_weight=3.0),
              drop_remainder=False),
          evaluation=Evaluation(
              ignored_label=ignore_label,
              max_instances_per_category=256,
              offset=256*256*256,
              is_thing=is_thing,
              rescale_predictions=True,
              report_per_class_pq=False,
              report_per_class_iou=False,
              report_train_mean_iou=False)),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=validation_steps,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.001,
                      'decay_steps': train_steps,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config


@exp_factory.register_config_factory('panoptic_deeplab_mobilenetv3_small_coco')
def panoptic_deeplab_mobilenetv3_small_coco() -> cfg.ExperimentConfig:
  """COCO panoptic segmentation with Panoptic Deeplab."""
  train_steps = 200000
  train_batch_size = 64
  eval_batch_size = 1
  steps_per_epoch = _COCO_TRAIN_EXAMPLES // train_batch_size
  validation_steps = _COCO_VAL_EXAMPLES // eval_batch_size

  num_panoptic_categories = 201
  num_thing_categories = 91
  ignore_label = 0

  is_thing = [False]
  for idx in range(1, num_panoptic_categories):
    is_thing.append(True if idx <= num_thing_categories else False)

  input_size = [640, 640, 3]
  output_stride = 16
  aspp_dilation_rates = [6, 12, 18]
  level = int(math.log2(output_stride))

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          mixed_precision_dtype='float32', enable_xla=True),
      task=PanopticDeeplabTask(
          init_checkpoint='gs://tf_model_garden/vision/panoptic/panoptic_deeplab/imagenet/mobilenetv3_small/ckpt-312000',
          init_checkpoint_modules=['backbone'],
          model=PanopticDeeplab(
              num_classes=num_panoptic_categories,
              input_size=input_size,
              backbone=backbones.Backbone(
                  type='mobilenet', mobilenet=backbones.MobileNet(
                      model_id='MobileNetV3Small',
                      filter_size_scale=1.0,
                      stochastic_depth_drop_rate=0.0,
                      output_stride=output_stride)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=decoders.ASPP(
                      level=level,
                      num_filters=256,
                      pool_kernel_size=input_size[:2],
                      dilation_rates=aspp_dilation_rates,
                      use_depthwise_convolution=True,
                      dropout_rate=0.1)),
              semantic_head=SemanticHead(
                  level=level,
                  num_convs=1,
                  num_filters=256,
                  kernel_size=5,
                  use_depthwise_convolution=True,
                  upsample_factor=1,
                  low_level=[3, 2],
                  low_level_num_filters=[64, 32],
                  fusion_num_output_filters=256,
                  prediction_kernel_size=1),
              instance_head=InstanceHead(
                  level=level,
                  num_convs=1,
                  num_filters=32,
                  kernel_size=5,
                  use_depthwise_convolution=True,
                  upsample_factor=1,
                  low_level=[3, 2],
                  low_level_num_filters=[32, 16],
                  fusion_num_output_filters=128,
                  prediction_kernel_size=1),
              shared_decoder=False,
              generate_panoptic_masks=True,
              post_processor=PanopticDeeplabPostProcessor(
                  output_size=input_size[:2],
                  center_score_threshold=0.1,
                  thing_class_ids=list(range(1, num_thing_categories)),
                  label_divisor=256,
                  stuff_area_limit=4096,
                  ignore_label=ignore_label,
                  nms_kernel=41,
                  keep_k_centers=200,
                  rescale_predictions=True)),
          losses=Losses(
              label_smoothing=0.0,
              ignore_label=ignore_label,
              l2_weight_decay=0.0,
              top_k_percent_pixels=0.2,
              segmentation_loss_weight=1.0,
              center_heatmap_loss_weight=200,
              center_offset_loss_weight=0.01),
          train_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_scale_min=0.5,
                  aug_scale_max=2.0,
                  aug_rand_hflip=True,
                  aug_type=common.Augmentation(
                      type='autoaug',
                      autoaug=common.AutoAugment(
                          augmentation_name='panoptic_deeplab_policy')),
                  sigma=8.0,
                  small_instance_area_threshold=4096,
                  small_instance_weight=3.0)),
          validation_data=DataConfig(
              input_path=os.path.join(_COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              parser=Parser(
                  resize_eval_groundtruth=False,
                  groundtruth_padded_size=[640, 640],
                  aug_scale_min=1.0,
                  aug_scale_max=1.0,
                  aug_rand_hflip=False,
                  aug_type=None,
                  sigma=8.0,
                  small_instance_area_threshold=4096,
                  small_instance_weight=3.0),
              drop_remainder=False),
          evaluation=Evaluation(
              ignored_label=ignore_label,
              max_instances_per_category=256,
              offset=256*256*256,
              is_thing=is_thing,
              rescale_predictions=True,
              report_per_class_pq=False,
              report_per_class_iou=False,
              report_train_mean_iou=False)),
      trainer=cfg.TrainerConfig(
          train_steps=train_steps,
          validation_steps=validation_steps,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adam',
              },
              'learning_rate': {
                  'type': 'polynomial',
                  'polynomial': {
                      'initial_learning_rate': 0.001,
                      'decay_steps': train_steps,
                      'end_learning_rate': 0.0,
                      'power': 0.9
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 2000,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config
