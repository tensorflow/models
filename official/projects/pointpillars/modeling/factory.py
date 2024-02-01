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

"""Factory methods to build models."""

from typing import Mapping, Optional

from absl import logging
import tensorflow as tf, tf_keras

from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.modeling import backbones
from official.projects.pointpillars.modeling import decoders
from official.projects.pointpillars.modeling import featurizers
from official.projects.pointpillars.modeling import heads
from official.projects.pointpillars.modeling import models
from official.vision.modeling.layers import detection_generator


def build_pointpillars(
    input_specs: Mapping[str, tf_keras.layers.InputSpec],
    model_config: cfg.PointPillarsModel,
    train_batch_size: int,
    eval_batch_size: int,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> tf_keras.Model:
  """Build the PointPillars model.

  Args:
    input_specs: A {name: input_spec} dict used to construct inputs.
    model_config: A PointPillarsModel config.
    train_batch_size: An `int` of training batch size per replica.
    eval_batch_size: An `int` of evaluation batch size per replica.
    l2_regularizer: A L2 regularizer.

  Returns:
    model: A PointPillarsModel built from the config.
  """
  # Build inputs
  inputs = {}
  for k, v in input_specs.items():
    inputs[k] = tf_keras.Input(shape=v.shape[1:], dtype=v.dtype)

  # Build featurizer
  image_size = (model_config.image.height, model_config.image.width)
  pillars_size = input_specs['pillars'].shape[1:]
  featurizer_config = model_config.featurizer
  featurizer = featurizers.Featurizer(
      image_size=image_size,
      pillars_size=pillars_size,
      num_blocks=featurizer_config.num_blocks,
      num_channels=featurizer_config.num_channels,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      kernel_regularizer=l2_regularizer)
  image = featurizer(inputs['pillars'], inputs['indices'], training=True)

  # Build backbone
  backbone_config = model_config.backbone
  backbone = backbones.Backbone(
      input_specs=featurizer.output_specs,
      min_level=backbone_config.min_level,
      max_level=backbone_config.max_level,
      num_convs=backbone_config.num_convs,
      kernel_regularizer=l2_regularizer)
  encoded_feats = backbone(image)

  # Build decoder
  decoder = decoders.Decoder(
      input_specs=backbone.output_specs,
      kernel_regularizer=l2_regularizer)
  decoded_feats = decoder(encoded_feats)

  # Build detection head
  head_config = model_config.head
  num_anchors_per_location = (len(model_config.anchors))
  head = heads.SSDHead(
      num_classes=model_config.num_classes,
      num_anchors_per_location=num_anchors_per_location,
      num_params_per_anchor=4,
      attribute_heads=[
          attr.as_dict() for attr in (head_config.attribute_heads or [])
      ],
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      kernel_regularizer=l2_regularizer)
  scores, boxes, attrs = head(decoded_feats)

  generator_config = model_config.detection_generator
  detection_generator_obj = detection_generator.MultilevelDetectionGenerator(
      apply_nms=generator_config.apply_nms,
      pre_nms_top_k=generator_config.pre_nms_top_k,
      pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
      nms_iou_threshold=generator_config.nms_iou_threshold,
      max_num_detections=generator_config.max_num_detections,
      nms_version=generator_config.nms_version,
      use_cpu_nms=generator_config.use_cpu_nms)

  image_size = [model_config.image.height, model_config.image.width]
  anchor_sizes = [(a.length, a.width) for a in model_config.anchors]
  model = models.PointPillarsModel(
      featurizer=featurizer,
      backbone=backbone,
      decoder=decoder,
      head=head,
      detection_generator=detection_generator_obj,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      image_size=image_size,
      anchor_sizes=anchor_sizes)

  logging.info('Train/Eval batch size per replica: %d/%d', train_batch_size,
               eval_batch_size)
  logging.info('Model inputs: %s', inputs)
  logging.info('Outputs in training:')
  logging.info('Featurizer output: %s', image)
  logging.info('Backbone output: %s', encoded_feats)
  logging.info('Decoder output: %s', decoded_feats)
  logging.info('Detection head outputs: scores %s, boxes %s, atrributes %s',
               scores, boxes, attrs)
  return model
