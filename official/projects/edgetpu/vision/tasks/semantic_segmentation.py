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

"""Image segmentation task definition."""
from typing import Any, Mapping, Optional

from absl import logging
import tensorflow as tf

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import task_factory
from official.projects.edgetpu.vision.configs import semantic_segmentation_config as exp_cfg
from official.projects.edgetpu.vision.configs import semantic_segmentation_searched_config as searched_cfg
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v1_model
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v2_model
from official.projects.edgetpu.vision.modeling.backbones import mobilenet_edgetpu  # pylint: disable=unused-import
from official.projects.edgetpu.vision.modeling.heads import bifpn_head
from official.vision.beta.dataloaders import input_reader_factory
from official.vision.beta.dataloaders import segmentation_input
from official.vision.beta.dataloaders import tfds_factory
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.tasks import semantic_segmentation


class ClassMappingParser(segmentation_input.Parser):
  """Same parser but maps classes max_class+1... to class 0."""

  max_class = 31

  def _prepare_image_and_label(self, data):
    """Prepare normalized image and label."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    label = tf.io.decode_image(data['image/segmentation/class/encoded'],
                               channels=1)
    height = data['image/height']
    width = data['image/width']
    image = tf.reshape(image, (height, width, 3))

    label = tf.reshape(label, (1, height, width))
    label = tf.where(
        tf.math.greater(label, self.max_class), tf.zeros_like(label), label)
    label = tf.where(tf.math.equal(label, 0), tf.ones_like(label)*255, label)
    label = tf.cast(label, tf.float32)
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(
        image, offset=[0.5, 0.5, 0.5], scale=[0.5, 0.5, 0.5])
    return image, label


@task_factory.register_task_cls(exp_cfg.CustomSemanticSegmentationTaskConfig)
class CustomSemanticSegmentationTask(
    semantic_segmentation.SemanticSegmentationTask):
  """A task for semantic segmentation."""

  def build_inputs(self,
                   params: cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds classification input."""
    ignore_label = self.task_config.losses.ignore_label

    if params.tfds_name:
      decoder = tfds_factory.get_segmentation_decoder(params.tfds_name)
    else:
      decoder = segmentation_input.Decoder()

    parser = ClassMappingParser(
        output_size=params.output_size,
        crop_size=params.crop_size,
        ignore_label=ignore_label,
        resize_eval_groundtruth=params.resize_eval_groundtruth,
        groundtruth_padded_size=params.groundtruth_padded_size,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        aug_rand_hflip=params.aug_rand_hflip,
        dtype=params.dtype)

    parser.max_class = self.task_config.model.num_classes-1

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset


class AutosegEdgeTPU(tf.keras.Model):
  """Segmentation keras network without pre/post-processing."""

  def __init__(self,
               model_params,
               min_level=3,
               max_level=8,
               output_filters=96,
               model_config=None,
               use_original_backbone_features=False,
               is_training_bn=True,
               strategy='tpu',
               data_format='channels_last',
               pooling_type='avg',
               fpn_num_filters=96,
               apply_bn_for_resampling=True,
               conv_after_downsample=True,
               upsampling_type='bilinear',
               conv_bn_act_pattern=True,
               conv_type='sep_3',
               head_conv_type='sep_3',
               act_type='relu6',
               fpn_weight_method='sum',
               output_weight_method='sum',
               fullres_output=False,
               num_classes=32,
               name='autoseg_edgetpu'):
    """Initialize model."""
    super().__init__()
    self.min_level = min_level
    self.max_level = max_level
    self.use_original_backbone_features = use_original_backbone_features
    self.strategy = strategy
    self.data_format = data_format
    model_name = model_params['model_name']
    self.backbone = get_models()[model_name](**model_params)

    # Feature network.
    self.resample_layers = []  # additional resampling layers.
    if use_original_backbone_features:
      start_level = 6
    else:
      # Not using original backbone features will (1) Use convolutions to
      # process all backbone features before feeding into FPN. (2) Use an extra
      # convolution to get higher level features, while preserve the channel
      # size from the last layer of backbone.
      start_level = min_level
      self.downsample_layers = []
      for level in range(start_level, max_level + 1):
        self.downsample_layers.append(
            bifpn_head.ResampleFeatureMap(
                feat_level=(level - min_level),
                target_num_channels=fpn_num_filters,
                is_training_bn=is_training_bn,
                strategy=strategy,
                data_format=data_format,
                pooling_type=pooling_type,
                name='downsample_p%d' % level,
            ))
    for level in range(start_level, max_level + 1):
      # Adds a coarser level by downsampling the last feature map.
      self.resample_layers.append(
          bifpn_head.ResampleFeatureMap(
              feat_level=(level - min_level),
              target_num_channels=fpn_num_filters,
              apply_bn=apply_bn_for_resampling,
              is_training_bn=is_training_bn,
              conv_after_downsample=conv_after_downsample,
              strategy=strategy,
              data_format=data_format,
              pooling_type=pooling_type,
              upsampling_type=upsampling_type,
              name='resample_p%d' % level,
          ))
    self.fpn_cells = bifpn_head.FPNCells(
        min_level=min_level,
        max_level=max_level,
        fpn_num_filters=fpn_num_filters,
        apply_bn_for_resampling=apply_bn_for_resampling,
        is_training_bn=is_training_bn,
        conv_after_downsample=conv_after_downsample,
        conv_bn_act_pattern=conv_bn_act_pattern,
        conv_type=conv_type,
        act_type=act_type,
        strategy=strategy,
        fpn_weight_method=fpn_weight_method,
        data_format=data_format,
        pooling_type=pooling_type,
        upsampling_type=upsampling_type,
        fpn_name='bifpn')

    self.seg_class_net = bifpn_head.SegClassNet(
        min_level=min_level,
        max_level=max_level,
        output_filters=output_filters,
        apply_bn_for_resampling=apply_bn_for_resampling,
        is_training_bn=is_training_bn,
        conv_after_downsample=conv_after_downsample,
        conv_bn_act_pattern=conv_bn_act_pattern,
        head_conv_type=head_conv_type,
        act_type=act_type,
        strategy=strategy,
        output_weight_method=output_weight_method,
        data_format=data_format,
        pooling_type=pooling_type,
        upsampling_type=upsampling_type,
        fullres_output=fullres_output,
        num_classes=num_classes)

  def call(self, inputs, training):
    # call backbone network.
    all_feats = self.backbone(inputs, training=training)
    if self.use_original_backbone_features:
      feats = all_feats[self.min_level:self.max_level + 1]
      for resample_layer in self.resample_layers:
        feats.append(resample_layer(feats[-1], training, None))
    else:
      feats = []
      for downsample_layer in self.downsample_layers:
        all_feats.append(downsample_layer(all_feats[-1], training, None))
      for level in range(self.min_level - 1, self.max_level):
        feats.append(self.resample_layers[level - self.min_level + 1](
            all_feats[level], training, all_feats[self.min_level - 1:]))

    # call feature network.
    feats = self.fpn_cells(feats, training)

    # call class/box output network.
    class_outputs = self.seg_class_net(feats, all_feats, training)

    return class_outputs


def get_models() -> Mapping[str, tf.keras.Model]:
  """Returns the mapping from model type name to Keras model."""
  model_mapping = {}

  def add_models(name: str, constructor: Any):
    if name in model_mapping:
      raise ValueError(f'Model {name} already exists in the mapping.')
    model_mapping[name] = constructor

  for model in mobilenet_edgetpu_v1_model.MODEL_CONFIGS.keys():
    add_models(model, mobilenet_edgetpu_v1_model.MobilenetEdgeTPU.from_name)

  for model in mobilenet_edgetpu_v2_model.MODEL_CONFIGS.keys():
    add_models(model, mobilenet_edgetpu_v2_model.MobilenetEdgeTPUV2.from_name)

  return model_mapping


@task_factory.register_task_cls(searched_cfg.AutosegEdgeTPUTaskConfig)
class AutosegEdgeTPUTask(semantic_segmentation.SemanticSegmentationTask):
  """A task for training the AutosegEdgeTPU models."""

  def build_model(self):
    """Builds model for training task."""
    model_config = self.task_config.model
    model_params = model_config.model_params.as_dict()
    model = AutosegEdgeTPU(
        model_params,
        min_level=model_config.head.min_level,
        max_level=model_config.head.max_level,
        fpn_num_filters=model_config.head.fpn_num_filters,
        num_classes=model_config.num_classes)
    logging.info(model_params)
    return model

  # TODO(suyoggupta): Dedup this function across tasks.
  def build_inputs(self,
                   params: cfg.DataConfig,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """Builds inputs for the segmentation task."""
    ignore_label = self.task_config.losses.ignore_label

    if params.tfds_name:
      decoder = tfds_factory.get_segmentation_decoder(params.tfds_name)
    else:
      decoder = segmentation_input.Decoder()

    parser = ClassMappingParser(
        output_size=params.output_size,
        crop_size=params.crop_size,
        ignore_label=ignore_label,
        resize_eval_groundtruth=params.resize_eval_groundtruth,
        groundtruth_padded_size=params.groundtruth_padded_size,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        aug_rand_hflip=params.aug_rand_hflip,
        dtype=params.dtype)

    parser.max_class = self.task_config.model.num_classes - 1

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)
    return dataset
