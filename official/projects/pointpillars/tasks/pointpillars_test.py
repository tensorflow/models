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

"""Tests for pointpillars."""

from absl.testing import parameterized
import tensorflow as tf

from official.core import exp_factory
from official.modeling import optimization
from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.tasks import pointpillars


def _mock_inputs(model_config):
  batch_size = 1
  image_config = model_config.image
  pillars_config = model_config.pillars
  pillars = tf.ones([
      batch_size, pillars_config.num_pillars,
      pillars_config.num_points_per_pillar,
      pillars_config.num_features_per_point
  ], dtype=tf.float32)
  indices = tf.ones([
      batch_size, pillars_config.num_pillars, 2
  ], dtype=tf.int32)
  features = {
      'pillars': pillars,
      'indices': indices,
  }

  image_height = image_config.height
  image_width = image_config.width
  num_anchors_per_location = len(model_config.anchors)
  cls_targets = {}
  box_targets = {}
  attribute_targets = {}
  for attr in model_config.head.attribute_heads:
    attribute_targets[attr.name] = {}
  total_num_anchors = 0
  for level in range(model_config.min_level, model_config.max_level + 1):
    stride = 2**level
    h_i = int(image_height / stride)
    w_i = int(image_width / stride)
    cls_targets[str(level)] = tf.ones(
        [batch_size, h_i, w_i, num_anchors_per_location], dtype=tf.int32)
    box_targets[str(level)] = tf.ones(
        [batch_size, h_i, w_i, num_anchors_per_location * 4], dtype=tf.float32)
    for attr in model_config.head.attribute_heads:
      attribute_targets[attr.name][str(level)] = tf.ones(
          [batch_size, h_i, w_i, num_anchors_per_location], dtype=tf.float32)
    total_num_anchors += h_i * w_i * num_anchors_per_location
  cls_weights = tf.ones([batch_size, total_num_anchors], dtype=tf.float32)
  box_weights = tf.ones([batch_size, total_num_anchors], dtype=tf.float32)
  image_shape = tf.ones([batch_size, 2], dtype=tf.int32)
  labels = {
      'cls_targets': cls_targets,
      'box_targets': box_targets,
      'attribute_targets': attribute_targets,
      'cls_weights': cls_weights,
      'box_weights': box_weights,
      'anchor_boxes': None,
      'image_shape': image_shape,
  }
  return features, labels


class PointPillarsTaskTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True),
      (False),
  )
  def test_train_and_eval(self, is_training):
    exp_config = exp_factory.get_exp_config('pointpillars_baseline')
    task_config = exp_config.task
    # modify config to suit local testing
    task_config.model.image.height = 32
    task_config.model.image.width = 32
    task_config.model.pillars.num_pillars = 2
    task_config.model.pillars.num_points_per_pillar = 3
    task_config.model.pillars.num_features_per_point = 4
    task_config.model.anchors = [cfg.Anchor(length=2.1, width=1.2)]

    task_config.train_data.global_batch_size = 1
    task_config.train_data.shuffle_buffer_size = 2
    task_config.validation_data.global_batch_size = 1
    task_config.validation_data.shuffle_buffer_size = 2
    task_config.use_wod_metrics = False

    task = pointpillars.PointPillarsTask(task_config)
    inputs = _mock_inputs(task_config.model)
    model = task.build_model()
    opt_factory = optimization.OptimizerFactory(
        exp_config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    metrics = task.build_metrics(training=is_training)

    if is_training:
      logs = task.train_step(inputs, model, optimizer, metrics=metrics)
    else:
      logs = task.validation_step(inputs, model, metrics=metrics)
    self.assertIn('loss', logs)

if __name__ == '__main__':
  tf.test.main()
