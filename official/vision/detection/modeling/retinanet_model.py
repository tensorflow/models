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
"""Model defination for the RetinaNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from absl import logging
import tensorflow.compat.v2 as tf

from tensorflow.python.keras import backend
from official.vision.detection.dataloader import mode_keys
from official.vision.detection.modeling import base_model
from official.vision.detection.modeling import losses
from official.vision.detection.modeling import postprocess
from official.vision.detection.modeling.architecture import factory
from official.vision.detection.evaluation import factory as eval_factory


class COCOMetrics(object):
  # This is only a wrapper for COCO metric and works on for numpy array. So it
  # doesn't inherit from tf.keras.layers.Layer or tf.keras.metrics.Metric.

  def __init__(self, params):
    self._evaluator = eval_factory.evaluator_generator(params.eval)

  def update_state(self, y_true, y_pred):
    labels = tf.nest.map_structure(lambda x: x.numpy(), y_true)
    outputs = tf.nest.map_structure(lambda x: x.numpy(), y_pred)
    groundtruths = {}
    predictions = {}
    for key, val in outputs.items():
      if isinstance(val, tuple):
        val = np.concatenate(val)
      predictions[key] = val
    for key, val in labels.items():
      if isinstance(val, tuple):
        val = np.concatenate(val)
      groundtruths[key] = val
    self._evaluator.update(predictions, groundtruths)

  def result(self):
    return self._evaluator.evaluate()

  def reset_states(self):
    return self._evaluator.reset()


class RetinanetModel(base_model.Model):
  """RetinaNet model function."""

  def __init__(self, params):
    super(RetinanetModel, self).__init__(params)

    # For eval metrics.
    self._params = params

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._head_fn = factory.retinanet_head_generator(params.retinanet_head)

    # Loss function.
    self._cls_loss_fn = losses.RetinanetClassLoss(params.retinanet_loss)
    self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
    self._box_loss_weight = params.retinanet_loss.box_loss_weight
    self._keras_model = None

    # Predict function.
    self._generate_detections_fn = postprocess.GenerateOneStageDetections(
        params.postprocess)

    self._l2_weight_decay = params.train.l2_weight_decay
    self._transpose_input = params.train.transpose_input
    assert not self._transpose_input, 'Transpose input is not supportted.'
    # Input layer.
    input_shape = (
        params.retinanet_parser.output_size +
        [params.retinanet_parser.num_channels])
    self._input_layer = tf.keras.layers.Input(
        shape=input_shape, name='',
        dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

  def build_outputs(self, inputs, mode):
    backbone_features = self._backbone_fn(
        inputs, is_training=(mode == mode_keys.TRAIN))
    fpn_features = self._fpn_fn(
        backbone_features, is_training=(mode == mode_keys.TRAIN))
    cls_outputs, box_outputs = self._head_fn(
        fpn_features, is_training=(mode == mode_keys.TRAIN))

    if self._use_bfloat16:
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

    model_outputs = {
        'cls_outputs': cls_outputs,
        'box_outputs': box_outputs,
    }
    return model_outputs

  def build_loss_fn(self):
    if self._keras_model is None:
      raise ValueError('build_loss_fn() must be called after build_model().')

    def _total_loss_fn(labels, outputs):
      cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                   labels['cls_targets'],
                                   labels['num_positives'])
      box_loss = self._box_loss_fn(outputs['box_outputs'],
                                   labels['box_targets'],
                                   labels['num_positives'])
      model_loss = cls_loss + self._box_loss_weight * box_loss
      l2_regularization_loss = self.weight_decay_loss(self._l2_weight_decay,
                                                      self._keras_model)
      total_loss = model_loss + l2_regularization_loss
      return {
          'total_loss': total_loss,
          'cls_loss': cls_loss,
          'box_loss': box_loss,
          'model_loss': model_loss,
          'l2_regularization_loss': l2_regularization_loss,
      }

    return _total_loss_fn

  def build_model(self, params, mode=None):
    if self._keras_model is None:
      with backend.get_graph().as_default():
        outputs = self.model_outputs(self._input_layer, mode)

        model = tf.keras.models.Model(
            inputs=self._input_layer, outputs=outputs, name='retinanet')
        assert model is not None, 'Fail to build tf.keras.Model.'
        model.optimizer = self.build_optimizer()
        self._keras_model = model

    return self._keras_model

  def post_processing(self, labels, outputs):
    required_output_fields = ['cls_outputs', 'box_outputs']
    for field in required_output_fields:
      if field not in outputs:
        raise ValueError('"%s" is missing in outputs, requried %s found %s',
                         field, required_output_fields, outputs.keys())
    required_label_fields = ['image_info', 'groundtruths']
    for field in required_label_fields:
      if field not in labels:
        raise ValueError('"%s" is missing in outputs, requried %s found %s',
                         field, required_label_fields, labels.keys())
    boxes, scores, classes, valid_detections = self._generate_detections_fn(
        inputs=(outputs['box_outputs'], outputs['cls_outputs'],
                labels['anchor_boxes'], labels['image_info'][:, 1:2, :]))
    # Discards the old output tensors to save memory. The `cls_outputs` and
    # `box_outputs` are pretty big and could potentiall lead to memory issue.
    outputs = {
        'source_id': labels['groundtruths']['source_id'],
        'image_info': labels['image_info'],
        'num_detections': valid_detections,
        'detection_boxes': boxes,
        'detection_classes': classes,
        'detection_scores': scores,
    }

    if 'groundtruths' in labels:
      labels['source_id'] = labels['groundtruths']['source_id']
      labels['boxes'] = labels['groundtruths']['boxes']
      labels['classes'] = labels['groundtruths']['classes']
      labels['areas'] = labels['groundtruths']['areas']
      labels['is_crowds'] = labels['groundtruths']['is_crowds']

    return labels, outputs

  def eval_metrics(self):
    return COCOMetrics(self._params)
