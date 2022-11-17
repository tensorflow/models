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

"""Tests for wod_detection_evaluator."""

import tensorflow as tf

from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.utils import wod_detection_evaluator


def _model_config():
  image = cfg.ImageConfig()
  image.x_range = (-10.0, 10.0)
  image.y_range = (-10.0, 10.0)
  image.resolution = 1.0
  image.height = 10
  image.width = 10
  model = cfg.PointPillarsModel()
  model.classes = 'all'
  model.image = image
  return model


def _create_boxes_1():
  boxes = tf.convert_to_tensor([[[1.1, 2.15, 3.2, 4.25], [0.3, 3.2, 3.4, 5.3],
                                 [0.4, 1.3, 3.5, 4.6], [2.6, 4.5, 4.7, 5.8]]],
                               dtype=tf.float32)
  classes = tf.convert_to_tensor([[1, 3, 1, 2]], dtype=tf.int64)
  attributes = {
      'heading':
          tf.convert_to_tensor([[[0.1], [0.2], [0.3], [0.4]]],
                               dtype=tf.float32),
      'z':
          tf.convert_to_tensor([[[1.1], [2.2], [3.3], [4.4]]],
                               dtype=tf.float32),
      'height':
          tf.convert_to_tensor([[[2.1], [2.2], [1.3], [1.4]]],
                               dtype=tf.float32),
  }
  return 4, boxes, classes, attributes


def _create_boxes_2():
  boxes = tf.convert_to_tensor([[[0.3, 0.5, 1.1, 2.05], [0.1, 2.4, 3.1, 4.2],
                                 [1.8, 2.1, 3.3, 4.3], [2.7, 3.2, 4.1, 5.9]]],
                               dtype=tf.float32)
  classes = tf.convert_to_tensor([[3, 2, 1, 3]], dtype=tf.int64)
  attributes = {
      'heading':
          tf.convert_to_tensor([[[1.4], [2.3], [3.2], [4.1]]],
                               dtype=tf.float32),
      'z':
          tf.convert_to_tensor([[[4.4], [3.3], [2.2], [1.1]]],
                               dtype=tf.float32),
      'height':
          tf.convert_to_tensor([[[3.4], [5.3], [6.2], [7.1]]],
                               dtype=tf.float32),
  }
  return 4, boxes, classes, attributes


def _create_y_true_1():
  n, boxes, classes, attributes = _create_boxes_1()
  frame_id = 1234
  y_true = {
      'frame_id': tf.convert_to_tensor([frame_id], dtype=tf.int64),
      'num_detections': tf.convert_to_tensor([n], dtype=tf.int64),
      'boxes': boxes,
      'classes': classes,
      'attributes': attributes,
      'difficulty': tf.convert_to_tensor([[1, 2, 2, 1]], dtype=tf.int64)
  }
  return y_true


def _create_y_pred_1():
  n, boxes, classes, attributes = _create_boxes_1()
  y_pred = {
      'num_detections': tf.convert_to_tensor([n], dtype=tf.int64),
      'boxes': boxes,
      'classes': classes,
      'attributes': attributes,
      'scores':
          tf.convert_to_tensor([[0.95, 0.9, 0.9, 0.9]], dtype=tf.float32)
  }
  return y_pred


def _create_y_pred_2():
  n, boxes, classes, attributes = _create_boxes_2()
  y_pred = {
      'num_detections': tf.convert_to_tensor([n], dtype=tf.int64),
      'boxes': boxes,
      'classes': classes,
      'attributes': attributes,
      'scores':
          tf.convert_to_tensor([[0.5, 0.2, 0.9, 0.9]], dtype=tf.float32)
  }
  return y_pred


class Wod3dDetectionEvaluatorTest(tf.test.TestCase):

  def test_wod_detection_evaluator(self):
    metric = wod_detection_evaluator.Wod3dDetectionEvaluator(_model_config())

    # Use perfect predictions
    y_pred, y_true = _create_y_pred_1(), _create_y_true_1()
    metric.update_state(groundtruths=y_true, predictions=y_pred)
    result = metric.result()
    for k, v in result.items():
      # Skip long range breakdown.
      if '[30, 50)' in k or '[50, +inf)' in k:
        continue
      self.assertAlmostEqual(v.numpy(), 1.0, places=3)

    # Use totally wrong predictions
    y_pred, y_true = _create_y_pred_2(), _create_y_true_1()
    metric.update_state(groundtruths=y_true, predictions=y_pred)
    result = metric.result()
    for k, v in result.items():
      self.assertAlmostEqual(v.numpy(), 0.0, places=3)


class Wod2dDetectionEvaluatorTest(tf.test.TestCase):

  def test_wod_detection_evaluator(self):
    metric = wod_detection_evaluator.Wod2dDetectionEvaluator(_model_config())

    # Use perfect predictions
    y_pred, y_true = _create_y_pred_1(), _create_y_true_1()
    metric.update_state(groundtruths=y_true, predictions=y_pred)
    result = metric.result()
    for k, v in result.items():
      # Skip long range breakdown.
      if '[30, 50)' in k or '[50, +inf)' in k:
        continue
      self.assertAlmostEqual(v.numpy(), 1.0, places=3)

    # Use totally wrong predictions
    y_pred, y_true = _create_y_pred_2(), _create_y_true_1()
    metric.update_state(groundtruths=y_true, predictions=y_pred)
    result = metric.result()
    for k, v in result.items():
      self.assertAlmostEqual(v.numpy(), 0.0, places=3)


if __name__ == '__main__':
  tf.test.main()
