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

"""Tests for detection_generator.py."""
# Import libraries

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.layers import detection_generator
from official.vision.beta.ops import anchor


class SelectTopKScoresTest(tf.test.TestCase):

  def testSelectTopKScores(self):
    pre_nms_num_boxes = 2
    scores_data = [[[0.2, 0.2], [0.1, 0.9], [0.5, 0.1], [0.3, 0.5]]]
    scores_in = tf.constant(scores_data, dtype=tf.float32)
    top_k_scores, top_k_indices = detection_generator._select_top_k_scores(
        scores_in, pre_nms_num_detections=pre_nms_num_boxes)
    expected_top_k_scores = np.array([[[0.5, 0.9], [0.3, 0.5]]],
                                     dtype=np.float32)

    expected_top_k_indices = [[[2, 1], [3, 3]]]

    self.assertAllEqual(top_k_scores.numpy(), expected_top_k_scores)
    self.assertAllEqual(top_k_indices.numpy(), expected_top_k_indices)


class DetectionGeneratorTest(
    parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True),
      (False),
  )
  def testDetectionsOutputShape(self, use_batched_nms):
    max_num_detections = 100
    num_classes = 4
    pre_nms_top_k = 5000
    pre_nms_score_threshold = 0.01
    batch_size = 1
    kwargs = {
        'apply_nms': True,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': 0.5,
        'max_num_detections': max_num_detections,
        'use_batched_nms': use_batched_nms,
    }
    generator = detection_generator.DetectionGenerator(**kwargs)

    cls_outputs_all = (
        np.random.rand(84, num_classes) - 0.5) * 3  # random 84x3 outputs.
    box_outputs_all = np.random.rand(84, 4 * num_classes)  # random 84 boxes.
    anchor_boxes_all = np.random.rand(84, 4)  # random 84 boxes.
    class_outputs = tf.reshape(
        tf.convert_to_tensor(cls_outputs_all, dtype=tf.float32),
        [1, 84, num_classes])
    box_outputs = tf.reshape(
        tf.convert_to_tensor(box_outputs_all, dtype=tf.float32),
        [1, 84, 4 * num_classes])
    anchor_boxes = tf.reshape(
        tf.convert_to_tensor(anchor_boxes_all, dtype=tf.float32),
        [1, 84, 4])
    image_info = tf.constant(
        [[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]],
        dtype=tf.float32)
    results = generator(
        box_outputs, class_outputs, anchor_boxes, image_info[:, 1, :])
    boxes = results['detection_boxes']
    classes = results['detection_classes']
    scores = results['detection_scores']
    valid_detections = results['num_detections']

    self.assertEqual(boxes.numpy().shape, (batch_size, max_num_detections, 4))
    self.assertEqual(scores.numpy().shape, (batch_size, max_num_detections,))
    self.assertEqual(classes.numpy().shape, (batch_size, max_num_detections,))
    self.assertEqual(valid_detections.numpy().shape, (batch_size,))

  def test_serialize_deserialize(self):
    kwargs = {
        'apply_nms': True,
        'pre_nms_top_k': 1000,
        'pre_nms_score_threshold': 0.1,
        'nms_iou_threshold': 0.5,
        'max_num_detections': 10,
        'use_batched_nms': False,
    }
    generator = detection_generator.DetectionGenerator(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(generator.get_config(), expected_config)

    new_generator = (
        detection_generator.DetectionGenerator.from_config(
            generator.get_config()))

    self.assertAllEqual(generator.get_config(), new_generator.get_config())


class MultilevelDetectionGeneratorTest(
    parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True),
      (False),
  )
  def testDetectionsOutputShape(self, use_batched_nms):
    min_level = 4
    max_level = 6
    num_scales = 2
    max_num_detections = 100
    aspect_ratios = [1.0, 2.0,]
    anchor_scale = 2.0
    output_size = [64, 64]
    num_classes = 4
    pre_nms_top_k = 5000
    pre_nms_score_threshold = 0.01
    batch_size = 1
    kwargs = {
        'apply_nms': True,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': 0.5,
        'max_num_detections': max_num_detections,
        'use_batched_nms': use_batched_nms,
    }

    input_anchor = anchor.build_anchor_generator(min_level, max_level,
                                                 num_scales, aspect_ratios,
                                                 anchor_scale)
    anchor_boxes = input_anchor(output_size)
    cls_outputs_all = (
        np.random.rand(84, num_classes) - 0.5) * 3  # random 84x3 outputs.
    box_outputs_all = np.random.rand(84, 4)  # random 84 boxes.
    class_outputs = {
        '4':
            tf.reshape(
                tf.convert_to_tensor(cls_outputs_all[0:64], dtype=tf.float32),
                [1, 8, 8, num_classes]),
        '5':
            tf.reshape(
                tf.convert_to_tensor(cls_outputs_all[64:80], dtype=tf.float32),
                [1, 4, 4, num_classes]),
        '6':
            tf.reshape(
                tf.convert_to_tensor(cls_outputs_all[80:84], dtype=tf.float32),
                [1, 2, 2, num_classes]),
    }
    box_outputs = {
        '4': tf.reshape(tf.convert_to_tensor(
            box_outputs_all[0:64], dtype=tf.float32), [1, 8, 8, 4]),
        '5': tf.reshape(tf.convert_to_tensor(
            box_outputs_all[64:80], dtype=tf.float32), [1, 4, 4, 4]),
        '6': tf.reshape(tf.convert_to_tensor(
            box_outputs_all[80:84], dtype=tf.float32), [1, 2, 2, 4]),
    }
    image_info = tf.constant([[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]],
                             dtype=tf.float32)
    generator = detection_generator.MultilevelDetectionGenerator(**kwargs)
    results = generator(box_outputs, class_outputs, anchor_boxes,
                        image_info[:, 1, :])
    boxes = results['detection_boxes']
    classes = results['detection_classes']
    scores = results['detection_scores']
    valid_detections = results['num_detections']

    self.assertEqual(boxes.numpy().shape, (batch_size, max_num_detections, 4))
    self.assertEqual(scores.numpy().shape, (batch_size, max_num_detections,))
    self.assertEqual(classes.numpy().shape, (batch_size, max_num_detections,))
    self.assertEqual(valid_detections.numpy().shape, (batch_size,))

  def test_serialize_deserialize(self):
    kwargs = {
        'apply_nms': True,
        'pre_nms_top_k': 1000,
        'pre_nms_score_threshold': 0.1,
        'nms_iou_threshold': 0.5,
        'max_num_detections': 10,
        'use_batched_nms': False,
    }
    generator = detection_generator.MultilevelDetectionGenerator(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(generator.get_config(), expected_config)

    new_generator = (
        detection_generator.MultilevelDetectionGenerator.from_config(
            generator.get_config()))

    self.assertAllEqual(generator.get_config(), new_generator.get_config())


if __name__ == '__main__':
  tf.test.main()
