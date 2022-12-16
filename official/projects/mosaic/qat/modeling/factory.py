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

"""Factory methods to build models."""
# Import libraries

import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.projects.mosaic.modeling import mosaic_blocks
from official.projects.mosaic.modeling import mosaic_head
from official.projects.mosaic.modeling import mosaic_model
from official.projects.mosaic.qat.modeling.heads import mosaic_head as qat_mosaic_head
from official.projects.mosaic.qat.modeling.layers import nn_blocks as qat_nn_blocks
from official.projects.qat.vision.configs import common
from official.projects.qat.vision.quantization import helper
from official.projects.qat.vision.quantization import schemes


def build_qat_mosaic_model(
    model: tf.keras.Model,
    quantization: common.Quantization,
    input_specs: tf.keras.layers.InputSpec) -> tf.keras.Model:
  """Applies quantization aware training for mosaic segmentation model.

  Args:
    model: The model applying quantization aware training.
    quantization: The Quantization config.
    input_specs: The shape specifications of input tensor.

  Returns:
    The model that applied optimization techniques.
  """

  original_checkpoint = quantization.pretrained_original_checkpoint
  if original_checkpoint is not None:
    ckpt = tf.train.Checkpoint(model=model, **model.checkpoint_items)
    status = ckpt.read(original_checkpoint)
    status.expect_partial().assert_existing_objects_matched()

  scope_dict = {
      'L2': tf.keras.regularizers.l2,
  }

  # Apply QAT to backbone (a tf.keras.Model) first, and then neck and head.
  with tfmot.quantization.keras.quantize_scope(scope_dict):
    annotated_backbone = tfmot.quantization.keras.quantize_annotate_model(
        model.backbone)
    optimized_backbone = tfmot.quantization.keras.quantize_apply(
        annotated_backbone, scheme=schemes.Default8BitQuantizeScheme())

    # Check for valid encoder and head.
    if not isinstance(model.head, mosaic_head.MosaicDecoderHead):
      raise ValueError('Only support MosaicDecoderHead for head.')
    if not isinstance(model.neck, mosaic_blocks.MosaicEncoderBlock):
      raise ValueError('Only support MosaicEncoderBlock for encoder.')

    head = qat_mosaic_head.MosaicDecoderHeadQuantized.from_config(
        model.head.get_config())
    neck = qat_nn_blocks.MosaicEncoderBlockQuantized.from_config(
        model.neck.get_config())

  optimized_model = mosaic_model.MosaicSegmentationModel(
      backbone=optimized_backbone,
      head=head,
      neck=neck)

  dummpy_input = tf.zeros([1] + list(input_specs.shape[1:]))
  optimized_model(dummpy_input, training=True)
  helper.copy_original_weights(model.head, optimized_model.head)
  helper.copy_original_weights(model.neck, optimized_model.neck)

  return optimized_model
