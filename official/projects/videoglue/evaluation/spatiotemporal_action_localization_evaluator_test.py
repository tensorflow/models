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

"""Tests for spatiotemporal_action_localization_evaluator."""
import tensorflow as tf

from official.projects.videoglue.evaluation import spatiotemporal_action_localization_evaluator as eval_util


class SpatiotemporalActionLocalizationEvaluatorTest(tf.test.TestCase):

  def _create_test_data_simple(self):
    boxes = tf.convert_to_tensor(
        [[[0.1, 0.15, 0.2, 0.25], [0.35, 0.18, 0.43, 0.4],
          [0.2, 0.1, 0.3, 0.2], [0.65, 0.55, 0.75, 0.85]],
         [[0.2, 0.5, 0.5, 0.8], [0.7, 0.1, 0.9, 0.9],
          [0.1, 0.4, 0.5, 0.7], [0.04, 0.05, 0.88, 0.77]]], dtype=tf.float32)
    nonmerge_boxes = boxes

    classes = tf.convert_to_tensor([[0, 2, 3, 4], [11, 12, 13, 14]],
                                   dtype=tf.int32)
    predictions = tf.one_hot(classes, depth=80)
    data = {
        'instances_position': boxes,
        'nonmerge_instances_position': nonmerge_boxes,
        'predictions': predictions,
        'nonmerge_label': classes,
    }
    return data

  def _create_test_data_complex(self):
    nonmerge_boxes = tf.convert_to_tensor(
        [[[0.1, 0.15, 0.2, 0.25], [0.1, 0.15, 0.2, 0.25],
          [0.2, 0.1, 0.3, 0.2], [0.65, 0.55, 0.75, 0.85]],
         [[0.2, 0.5, 0.5, 0.8], [0.7, 0.1, 0.9, 0.9],
          [0.2, 0.5, 0.5, 0.8], [0.7, 0.1, 0.9, 0.9]]], dtype=tf.float32)
    boxes = tf.convert_to_tensor(
        [[[0.1, 0.15, 0.2, 0.25], [0.2, 0.1, 0.3, 0.2],
          [0.65, 0.55, 0.75, 0.85], [-1, -1, -1, -1]],
         [[0.2, 0.5, 0.5, 0.8], [0.7, 0.1, 0.9, 0.9],
          [-1, -1, -1, -1], [-1, -1, -1, -1]]], dtype=tf.float32)

    classes = tf.convert_to_tensor([[0, 2, 3, 4], [11, 12, 13, 14]],
                                   dtype=tf.int32)
    predictions = tf.one_hot(classes, depth=80)

    data = {
        'instances_position': boxes,
        'nonmerge_instances_position': nonmerge_boxes,
        'predictions': predictions,
        'nonmerge_label': classes,
    }
    return data

  def test_action_localization_eval_simple(self):
    data = self._create_test_data_simple()
    evaluator = eval_util.SpatiotemporalActionLocalizationEvaluator()
    evaluator.reset_states()
    evaluator.update_state(data)
    metrics = evaluator.result()
    self.assertAlmostEqual(metrics['mAP@.5IOU'], 1.0)

  def test_action_localization_eval_complex(self):
    data = self._create_test_data_complex()
    evaluator = eval_util.SpatiotemporalActionLocalizationEvaluator()
    evaluator.reset_states()
    evaluator.update_state(data)
    metrics = evaluator.result()
    self.assertAlmostEqual(metrics['mAP@.5IOU'], 0.64375)


if __name__ == '__main__':
  tf.test.main()
