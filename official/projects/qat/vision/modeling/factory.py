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
from official.projects.qat.vision.configs import common
from official.projects.qat.vision.modeling import segmentation_model as qat_segmentation_model
from official.projects.qat.vision.n_bit import schemes as n_bit_schemes
from official.projects.qat.vision.quantization import schemes
from official.vision import configs
from official.vision.modeling import classification_model
from official.vision.modeling import retinanet_model
from official.vision.modeling.decoders import aspp
from official.vision.modeling.heads import segmentation_heads
from official.vision.modeling.layers import nn_layers


def build_qat_classification_model(
    model: tf.keras.Model,
    quantization: common.Quantization,
    input_specs: tf.keras.layers.InputSpec,
    model_config: configs.image_classification.ImageClassificationModel,
    l2_regularizer: tf.keras.regularizers.Regularizer = None
) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Apply model optimization techniques.

  Args:
    model: The model applying model optimization techniques.
    quantization: The Quantization config.
    input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
    model_config: The model config.
    l2_regularizer: tf.keras.regularizers.Regularizer object. Default to None.

  Returns:
    model: The model that applied optimization techniques.
  """
  original_checkpoint = quantization.pretrained_original_checkpoint
  if original_checkpoint:
    ckpt = tf.train.Checkpoint(
        model=model,
        **model.checkpoint_items)
    status = ckpt.read(original_checkpoint)
    status.expect_partial().assert_existing_objects_matched()

  scope_dict = {
      'L2': tf.keras.regularizers.l2,
  }
  with tfmot.quantization.keras.quantize_scope(scope_dict):
    annotated_backbone = tfmot.quantization.keras.quantize_annotate_model(
        model.backbone)
    if quantization.change_num_bits:
      backbone = tfmot.quantization.keras.quantize_apply(
          annotated_backbone,
          scheme=n_bit_schemes.DefaultNBitQuantizeScheme(
              num_bits_weight=quantization.num_bits_weight,
              num_bits_activation=quantization.num_bits_activation))
    else:
      backbone = tfmot.quantization.keras.quantize_apply(
          annotated_backbone,
          scheme=schemes.Default8BitQuantizeScheme())

  norm_activation_config = model_config.norm_activation
  backbone_optimized_model = classification_model.ClassificationModel(
      backbone=backbone,
      num_classes=model_config.num_classes,
      input_specs=input_specs,
      dropout_rate=model_config.dropout_rate,
      kernel_regularizer=l2_regularizer,
      add_head_batch_norm=model_config.add_head_batch_norm,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon)
  for from_layer, to_layer in zip(
      model.layers, backbone_optimized_model.layers):
    if from_layer != model.backbone:
      to_layer.set_weights(from_layer.get_weights())

  with tfmot.quantization.keras.quantize_scope(scope_dict):
    def apply_quantization_to_dense(layer):
      if isinstance(layer, (tf.keras.layers.Dense,
                            tf.keras.layers.Dropout,
                            tf.keras.layers.GlobalAveragePooling2D)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
      return layer

    annotated_model = tf.keras.models.clone_model(
        backbone_optimized_model,
        clone_function=apply_quantization_to_dense,
    )

    if quantization.change_num_bits:
      optimized_model = tfmot.quantization.keras.quantize_apply(
          annotated_model,
          scheme=n_bit_schemes.DefaultNBitQuantizeScheme(
              num_bits_weight=quantization.num_bits_weight,
              num_bits_activation=quantization.num_bits_activation))

    else:
      optimized_model = tfmot.quantization.keras.quantize_apply(
          annotated_model)

  return optimized_model


def build_qat_retinanet(
    model: tf.keras.Model, quantization: common.Quantization,
    model_config: configs.retinanet.RetinaNet) -> tf.keras.Model:
  """Applies quantization aware training for RetinaNet model.

  Args:
    model: The model applying quantization aware training.
    quantization: The Quantization config.
    model_config: The model config.

  Returns:
    The model that applied optimization techniques.
  """

  original_checkpoint = quantization.pretrained_original_checkpoint
  if original_checkpoint is not None:
    ckpt = tf.train.Checkpoint(
        model=model,
        **model.checkpoint_items)
    status = ckpt.read(original_checkpoint)
    status.expect_partial().assert_existing_objects_matched()

  scope_dict = {
      'L2': tf.keras.regularizers.l2,
  }
  with tfmot.quantization.keras.quantize_scope(scope_dict):
    annotated_backbone = tfmot.quantization.keras.quantize_annotate_model(
        model.backbone)
    optimized_backbone = tfmot.quantization.keras.quantize_apply(
        annotated_backbone,
        scheme=schemes.Default8BitQuantizeScheme())
  optimized_model = retinanet_model.RetinaNetModel(
      optimized_backbone,
      model.decoder,
      model.head,
      model.detection_generator,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_scales=model_config.anchor.num_scales,
      aspect_ratios=model_config.anchor.aspect_ratios,
      anchor_size=model_config.anchor.anchor_size)
  return optimized_model


def build_qat_segmentation_model(
    model: tf.keras.Model, quantization: common.Quantization,
    input_specs: tf.keras.layers.InputSpec) -> tf.keras.Model:
  """Applies quantization aware training for segmentation model.

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

  # Build quantization compatible model.
  model = qat_segmentation_model.SegmentationModelQuantized(
      model.backbone, model.decoder, model.head, input_specs)

  scope_dict = {
      'L2': tf.keras.regularizers.l2,
  }

  # Apply QAT to backbone (a tf.keras.Model) first.
  with tfmot.quantization.keras.quantize_scope(scope_dict):
    annotated_backbone = tfmot.quantization.keras.quantize_annotate_model(
        model.backbone)
    optimized_backbone = tfmot.quantization.keras.quantize_apply(
        annotated_backbone, scheme=schemes.Default8BitQuantizeScheme())
  backbone_optimized_model = qat_segmentation_model.SegmentationModelQuantized(
      optimized_backbone, model.decoder, model.head, input_specs)

  # Copy over all remaining layers.
  for from_layer, to_layer in zip(model.layers,
                                  backbone_optimized_model.layers):
    if from_layer != model.backbone:
      to_layer.set_weights(from_layer.get_weights())

  with tfmot.quantization.keras.quantize_scope(scope_dict):

    def apply_quantization_to_layers(layer):
      if isinstance(layer, (segmentation_heads.SegmentationHead,
                            nn_layers.SpatialPyramidPooling, aspp.ASPP)):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
      return layer

    annotated_model = tf.keras.models.clone_model(
        backbone_optimized_model,
        clone_function=apply_quantization_to_layers,
    )
    optimized_model = tfmot.quantization.keras.quantize_apply(
        annotated_model, scheme=schemes.Default8BitQuantizeScheme())

  return optimized_model
