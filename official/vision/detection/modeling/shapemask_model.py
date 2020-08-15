# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model definition for the ShapeMask Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.vision.detection.dataloader import anchor
from official.vision.detection.dataloader import mode_keys
from official.vision.detection.evaluation import factory as eval_factory
from official.vision.detection.modeling import base_model
from official.vision.detection.modeling import losses
from official.vision.detection.modeling.architecture import factory
from official.vision.detection.modeling.architecture import keras_utils
from official.vision.detection.ops import postprocess_ops
from official.vision.detection.utils import box_utils


class ShapeMaskModel(base_model.Model):
  """ShapeMask model function."""

  def __init__(self, params):
    super(ShapeMaskModel, self).__init__(params)

    self._params = params
    self._keras_model = None

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._retinanet_head_fn = factory.retinanet_head_generator(params)
    self._shape_prior_head_fn = factory.shapeprior_head_generator(params)
    self._coarse_mask_fn = factory.coarsemask_head_generator(params)
    self._fine_mask_fn = factory.finemask_head_generator(params)

    # Loss functions.
    self._cls_loss_fn = losses.RetinanetClassLoss(
        params.retinanet_loss, params.architecture.num_classes)
    self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
    self._box_loss_weight = params.retinanet_loss.box_loss_weight

    # Mask loss function.
    self._shapemask_prior_loss_fn = losses.ShapemaskMseLoss()
    self._shapemask_loss_fn = losses.ShapemaskLoss()
    self._shape_prior_loss_weight = (
        params.shapemask_loss.shape_prior_loss_weight)
    self._coarse_mask_loss_weight = (
        params.shapemask_loss.coarse_mask_loss_weight)
    self._fine_mask_loss_weight = (params.shapemask_loss.fine_mask_loss_weight)

    # Predict function.
    self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
        params.architecture.min_level, params.architecture.max_level,
        params.postprocess)

  def build_outputs(self, inputs, mode):
    is_training = mode == mode_keys.TRAIN
    images = inputs['image']

    if 'anchor_boxes' in inputs:
      anchor_boxes = inputs['anchor_boxes']
    else:
      anchor_boxes = anchor.Anchor(
          self._params.architecture.min_level,
          self._params.architecture.max_level, self._params.anchor.num_scales,
          self._params.anchor.aspect_ratios, self._params.anchor.anchor_size,
          images.get_shape().as_list()[1:3]).multilevel_boxes

      batch_size = tf.shape(images)[0]
      for level in anchor_boxes:
        anchor_boxes[level] = tf.tile(
            tf.expand_dims(anchor_boxes[level], 0), [batch_size, 1, 1, 1])

    backbone_features = self._backbone_fn(images, is_training=is_training)
    fpn_features = self._fpn_fn(backbone_features, is_training=is_training)
    cls_outputs, box_outputs = self._retinanet_head_fn(
        fpn_features, is_training=is_training)

    valid_boxes, valid_scores, valid_classes, valid_detections = (
        self._generate_detections_fn(box_outputs, cls_outputs, anchor_boxes,
                                     inputs['image_info'][:, 1:2, :]))

    image_size = images.get_shape().as_list()[1:3]
    valid_outer_boxes = box_utils.compute_outer_boxes(
        tf.reshape(valid_boxes, [-1, 4]),
        image_size,
        scale=self._params.shapemask_parser.outer_box_scale)
    valid_outer_boxes = tf.reshape(valid_outer_boxes, tf.shape(valid_boxes))

    # Wrapping if else code paths into a layer to make the checkpoint loadable
    # in prediction mode.
    class SampledBoxesLayer(tf.keras.layers.Layer):
      """ShapeMask model function."""

      def call(self, inputs, val_boxes, val_classes, val_outer_boxes, training):
        if training:
          boxes = inputs['mask_boxes']
          outer_boxes = inputs['mask_outer_boxes']
          classes = inputs['mask_classes']
        else:
          boxes = val_boxes
          classes = val_classes
          outer_boxes = val_outer_boxes
        return boxes, classes, outer_boxes

    boxes, classes, outer_boxes = SampledBoxesLayer()(
        inputs,
        valid_boxes,
        valid_classes,
        valid_outer_boxes,
        training=is_training)

    instance_features, prior_masks = self._shape_prior_head_fn(
        fpn_features, boxes, outer_boxes, classes, is_training)
    coarse_mask_logits = self._coarse_mask_fn(instance_features, prior_masks,
                                              classes, is_training)
    fine_mask_logits = self._fine_mask_fn(instance_features, coarse_mask_logits,
                                          classes, is_training)

    model_outputs = {
        'cls_outputs': cls_outputs,
        'box_outputs': box_outputs,
        'fine_mask_logits': fine_mask_logits,
        'coarse_mask_logits': coarse_mask_logits,
        'prior_masks': prior_masks,
    }

    if not is_training:
      model_outputs.update({
          'num_detections': valid_detections,
          'detection_boxes': valid_boxes,
          'detection_outer_boxes': valid_outer_boxes,
          'detection_masks': fine_mask_logits,
          'detection_classes': valid_classes,
          'detection_scores': valid_scores,
      })

    return model_outputs

  def build_loss_fn(self):
    if self._keras_model is None:
      raise ValueError('build_loss_fn() must be called after build_model().')

    filter_fn = self.make_filter_trainable_variables_fn()
    trainable_variables = filter_fn(self._keras_model.trainable_variables)

    def _total_loss_fn(labels, outputs):
      cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                   labels['cls_targets'],
                                   labels['num_positives'])
      box_loss = self._box_loss_fn(outputs['box_outputs'],
                                   labels['box_targets'],
                                   labels['num_positives'])

      # Adds Shapemask model losses.
      shape_prior_loss = self._shapemask_prior_loss_fn(outputs['prior_masks'],
                                                       labels['mask_targets'],
                                                       labels['mask_is_valid'])
      coarse_mask_loss = self._shapemask_loss_fn(outputs['coarse_mask_logits'],
                                                 labels['mask_targets'],
                                                 labels['mask_is_valid'])
      fine_mask_loss = self._shapemask_loss_fn(outputs['fine_mask_logits'],
                                               labels['fine_mask_targets'],
                                               labels['mask_is_valid'])

      model_loss = (
          cls_loss + self._box_loss_weight * box_loss +
          shape_prior_loss * self._shape_prior_loss_weight +
          coarse_mask_loss * self._coarse_mask_loss_weight +
          fine_mask_loss * self._fine_mask_loss_weight)

      l2_regularization_loss = self.weight_decay_loss(trainable_variables)
      total_loss = model_loss + l2_regularization_loss

      shapemask_losses = {
          'total_loss': total_loss,
          'loss': total_loss,
          'retinanet_cls_loss': cls_loss,
          'l2_regularization_loss': l2_regularization_loss,
          'retinanet_box_loss': box_loss,
          'shapemask_prior_loss': shape_prior_loss,
          'shapemask_coarse_mask_loss': coarse_mask_loss,
          'shapemask_fine_mask_loss': fine_mask_loss,
          'model_loss': model_loss,
      }
      return shapemask_losses

    return _total_loss_fn

  def build_input_layers(self, params, mode):
    is_training = mode == mode_keys.TRAIN
    input_shape = (
        params.shapemask_parser.output_size +
        [params.shapemask_parser.num_channels])
    if is_training:
      batch_size = params.train.batch_size
      input_layer = {
          'image':
              tf.keras.layers.Input(
                  shape=input_shape,
                  batch_size=batch_size,
                  name='image',
                  dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32),
          'image_info':
              tf.keras.layers.Input(
                  shape=[4, 2], batch_size=batch_size, name='image_info'),
          'mask_classes':
              tf.keras.layers.Input(
                  shape=[params.shapemask_parser.num_sampled_masks],
                  batch_size=batch_size,
                  name='mask_classes',
                  dtype=tf.int64),
          'mask_outer_boxes':
              tf.keras.layers.Input(
                  shape=[params.shapemask_parser.num_sampled_masks, 4],
                  batch_size=batch_size,
                  name='mask_outer_boxes',
                  dtype=tf.float32),
          'mask_boxes':
              tf.keras.layers.Input(
                  shape=[params.shapemask_parser.num_sampled_masks, 4],
                  batch_size=batch_size,
                  name='mask_boxes',
                  dtype=tf.float32),
      }
    else:
      batch_size = params.eval.batch_size
      input_layer = {
          'image':
              tf.keras.layers.Input(
                  shape=input_shape,
                  batch_size=batch_size,
                  name='image',
                  dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32),
          'image_info':
              tf.keras.layers.Input(
                  shape=[4, 2], batch_size=batch_size, name='image_info'),
      }
    return input_layer

  def build_model(self, params, mode):
    if self._keras_model is None:
      input_layers = self.build_input_layers(self._params, mode)
      with keras_utils.maybe_enter_backend_graph():
        outputs = self.model_outputs(input_layers, mode)

        model = tf.keras.models.Model(
            inputs=input_layers, outputs=outputs, name='shapemask')
        assert model is not None, 'Fail to build tf.keras.Model.'
        model.optimizer = self.build_optimizer()
        self._keras_model = model

    return self._keras_model

  def post_processing(self, labels, outputs):
    required_output_fields = [
        'num_detections', 'detection_boxes', 'detection_classes',
        'detection_masks', 'detection_scores'
    ]

    for field in required_output_fields:
      if field not in outputs:
        raise ValueError(
            '"{}" is missing in outputs, requried {} found {}'.format(
                field, required_output_fields, outputs.keys()))

    required_label_fields = ['image_info']
    for field in required_label_fields:
      if field not in labels:
        raise ValueError(
            '"{}" is missing in labels, requried {} found {}'.format(
                field, required_label_fields, labels.keys()))

    predictions = {
        'image_info': labels['image_info'],
        'num_detections': outputs['num_detections'],
        'detection_boxes': outputs['detection_boxes'],
        'detection_outer_boxes': outputs['detection_outer_boxes'],
        'detection_classes': outputs['detection_classes'],
        'detection_scores': outputs['detection_scores'],
        'detection_masks': outputs['detection_masks'],
    }

    if 'groundtruths' in labels:
      predictions['source_id'] = labels['groundtruths']['source_id']
      labels = labels['groundtruths']

    return labels, predictions

  def eval_metrics(self):
    return eval_factory.evaluator_generator(self._params.eval)
