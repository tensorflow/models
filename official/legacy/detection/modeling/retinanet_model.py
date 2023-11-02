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

"""Model defination for the RetinaNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, tf_keras

from official.legacy.detection.dataloader import mode_keys
from official.legacy.detection.evaluation import factory as eval_factory
from official.legacy.detection.modeling import base_model
from official.legacy.detection.modeling import losses
from official.legacy.detection.modeling.architecture import factory
from official.legacy.detection.ops import postprocess_ops


class RetinanetModel(base_model.Model):
  """RetinaNet model function."""

  def __init__(self, params):
    super(RetinanetModel, self).__init__(params)

    # For eval metrics.
    self._params = params

    # Architecture generators.
    self._backbone_fn = factory.backbone_generator(params)
    self._fpn_fn = factory.multilevel_features_generator(params)
    self._head_fn = factory.retinanet_head_generator(params)

    # Loss function.
    self._cls_loss_fn = losses.RetinanetClassLoss(
        params.retinanet_loss, params.architecture.num_classes)
    self._box_loss_fn = losses.RetinanetBoxLoss(params.retinanet_loss)
    self._box_loss_weight = params.retinanet_loss.box_loss_weight
    self._keras_model = None

    # Predict function.
    self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
        params.architecture.min_level, params.architecture.max_level,
        params.postprocess)

    self._transpose_input = params.train.transpose_input
    assert not self._transpose_input, 'Transpose input is not supported.'
    # Input layer.
    self._input_layer = tf_keras.layers.Input(
        shape=(None, None, params.retinanet_parser.num_channels),
        name='',
        dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

  def build_outputs(self, inputs, mode):
    # If the input image is transposed (from NHWC to HWCN), we need to revert it
    # back to the original shape before it's used in the computation.
    if self._transpose_input:
      inputs = tf.transpose(inputs, [3, 0, 1, 2])

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

    filter_fn = self.make_filter_trainable_variables_fn()
    trainable_variables = filter_fn(self._keras_model.trainable_variables)

    def _total_loss_fn(labels, outputs):
      cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                   labels['cls_targets'],
                                   labels['num_positives'])
      box_loss = self._box_loss_fn(outputs['box_outputs'],
                                   labels['box_targets'],
                                   labels['num_positives'])
      model_loss = cls_loss + self._box_loss_weight * box_loss
      l2_regularization_loss = self.weight_decay_loss(trainable_variables)
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
      outputs = self.model_outputs(self._input_layer, mode)

      model = tf_keras.models.Model(
          inputs=self._input_layer, outputs=outputs, name='retinanet')
      assert model is not None, 'Fail to build tf_keras.Model.'
      model.optimizer = self.build_optimizer()
      self._keras_model = model

    return self._keras_model

  def post_processing(self, labels, outputs):
    # TODO(yeqing): Moves the output related part into build_outputs.
    required_output_fields = ['cls_outputs', 'box_outputs']
    for field in required_output_fields:
      if field not in outputs:
        raise ValueError('"%s" is missing in outputs, requried %s found %s' %
                         (field, required_output_fields, outputs.keys()))
    required_label_fields = ['image_info', 'groundtruths']
    for field in required_label_fields:
      if field not in labels:
        raise ValueError('"%s" is missing in outputs, requried %s found %s' %
                         (field, required_label_fields, labels.keys()))
    boxes, scores, classes, valid_detections = self._generate_detections_fn(
        outputs['box_outputs'], outputs['cls_outputs'], labels['anchor_boxes'],
        labels['image_info'][:, 1:2, :])
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
    return eval_factory.evaluator_generator(self._params.eval)
