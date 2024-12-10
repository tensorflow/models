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

"""Tests for detection_generator.py."""
from unittest import mock

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.vision.configs import common
from official.vision.modeling.layers import detection_generator
from official.vision.ops import anchor


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

  @parameterized.product(
      nms_version=['batched', 'v1', 'v2'],
      use_cpu_nms=[True, False],
      soft_nms_sigma=[None, 0.1],
      use_sigmoid_probability=[True, False])
  def testDetectionsOutputShape(self, nms_version, use_cpu_nms, soft_nms_sigma,
                                use_sigmoid_probability):
    max_num_detections = 10
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
        'nms_version': nms_version,
        'use_cpu_nms': use_cpu_nms,
        'soft_nms_sigma': soft_nms_sigma,
        'use_sigmoid_probability': use_sigmoid_probability,
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
        'nms_version': 'v2',
        'use_cpu_nms': False,
        'soft_nms_sigma': None,
        'use_sigmoid_probability': False,
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
      ('batched', False, True, None, None, None),
      ('batched', False, False, None, None, None),
      ('v3', False, True, None, None, None),
      ('v3', False, False, None, None, None),
      ('v2', False, True, None, None, None),
      ('v2', False, False, None, None, None),
      ('v2', False, False, None, None, True),
      ('v1', True, True, 0.0, None, None),
      ('v1', True, False, 0.1, None, None),
      ('v1', True, False, None, None, None),
      ('tflite', False, False, None, True, None),
      ('tflite', False, False, None, False, None),
  )
  def testDetectionsOutputShape(
      self,
      nms_version,
      has_att_heads,
      use_cpu_nms,
      soft_nms_sigma,
      use_regular_nms,
      use_class_agnostic_nms,
  ):
    min_level = 4
    max_level = 6
    num_scales = 2
    max_num_detections = 10
    aspect_ratios = [1.0, 2.0]
    anchor_scale = 2.0
    output_size = [64, 64]
    num_classes = 4
    pre_nms_top_k = 5000
    pre_nms_score_threshold = 0.01
    batch_size = 1
    tflite_post_processing_config = {
        'max_detections': max_num_detections,
        'max_classes_per_detection': 1,
        'use_regular_nms': use_regular_nms,
        'nms_score_threshold': 0.01,
        'nms_iou_threshold': 0.5,
        'input_image_size': [224, 224],
        'normalize_anchor_coordinates': True,
    }
    kwargs = {
        'apply_nms': True,
        'pre_nms_top_k': pre_nms_top_k,
        'pre_nms_score_threshold': pre_nms_score_threshold,
        'nms_iou_threshold': 0.5,
        'max_num_detections': max_num_detections,
        'nms_version': nms_version,
        'use_cpu_nms': use_cpu_nms,
        'soft_nms_sigma': soft_nms_sigma,
        'tflite_post_processing_config': tflite_post_processing_config,
        'use_class_agnostic_nms': use_class_agnostic_nms,
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
    if has_att_heads:
      att_outputs_all = np.random.rand(84, 1)  # random attributes.
      att_outputs = {
          'depth': {
              '4':
                  tf.reshape(
                      tf.convert_to_tensor(
                          att_outputs_all[0:64], dtype=tf.float32),
                      [1, 8, 8, 1]),
              '5':
                  tf.reshape(
                      tf.convert_to_tensor(
                          att_outputs_all[64:80], dtype=tf.float32),
                      [1, 4, 4, 1]),
              '6':
                  tf.reshape(
                      tf.convert_to_tensor(
                          att_outputs_all[80:84], dtype=tf.float32),
                      [1, 2, 2, 1]),
          }
      }
    else:
      att_outputs = None
    image_info = tf.constant([[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]],
                             dtype=tf.float32)
    generator = detection_generator.MultilevelDetectionGenerator(**kwargs)
    results = generator(box_outputs, class_outputs, anchor_boxes,
                        image_info[:, 1, :], att_outputs)
    boxes = results['detection_boxes']
    classes = results['detection_classes']
    scores = results['detection_scores']
    valid_detections = results['num_detections']

    if nms_version == 'tflite':
      # When nms_version is `tflite`, all output tensors are empty as the actual
      # post-processing happens in the TFLite model.
      self.assertEqual(boxes.numpy().shape, ())
      self.assertEqual(scores.numpy().shape, ())
      self.assertEqual(classes.numpy().shape, ())
      self.assertEqual(valid_detections.numpy().shape, ())
    else:
      self.assertEqual(boxes.numpy().shape, (batch_size, max_num_detections, 4))
      self.assertEqual(scores.numpy().shape, (
          batch_size,
          max_num_detections,
      ))
      self.assertEqual(classes.numpy().shape, (
          batch_size,
          max_num_detections,
      ))
      self.assertEqual(valid_detections.numpy().shape, (batch_size,))
      if has_att_heads:
        for att in results['detection_attributes'].values():
          self.assertEqual(att.numpy().shape,
                           (batch_size, max_num_detections, 1))

  def test_decode_multilevel_outputs_and_pre_nms_top_k(self):
    named_params = {
        'apply_nms': True,
        'pre_nms_top_k': 5,
        'pre_nms_score_threshold': 0.05,
        'nms_iou_threshold': 0.5,
        'max_num_detections': 2,
        'nms_version': 'v3',
        'use_cpu_nms': False,
        'soft_nms_sigma': None,
    }
    generator = detection_generator.MultilevelDetectionGenerator(**named_params)
    # 2 classes, 3 boxes per pixel, 2 levels '1': 2x2, '2':1x1
    background = [1, 0, 0]
    first = [0, 1, 0]
    second = [0, 0, 1]
    some = [0, 0.5, 0.5]
    class_outputs = {
        '1':
            tf.constant([[[
                first + background + first, first + background + second
            ], [second + background + first, second + background + second]]],
                        dtype=tf.float32),
        '2':
            tf.constant([[[background + some + background]]], dtype=tf.float32),
    }
    box_outputs = {
        '1': tf.zeros(shape=[1, 2, 2, 12], dtype=tf.float32),
        '2': tf.zeros(shape=[1, 1, 1, 12], dtype=tf.float32)
    }
    anchor_boxes = {
        '1':
            tf.random.uniform(
                shape=[2, 2, 12], minval=1., maxval=99., dtype=tf.float32),
        '2':
            tf.random.uniform(
                shape=[1, 1, 12], minval=1., maxval=99., dtype=tf.float32),
    }
    boxes, scores = generator._decode_multilevel_outputs_and_pre_nms_top_k(
        box_outputs, class_outputs, anchor_boxes,
        tf.constant([[100, 100]], dtype=tf.float32))
    self.assertAllClose(
        scores,
        tf.sigmoid(
            tf.constant([[[1, 1, 1, 1, 0.5], [1, 1, 1, 1, 0.5]]],
                        dtype=tf.float32)))
    self.assertAllClose(
        tf.squeeze(boxes),
        tf.stack([
            # Where the first is + some as last
            tf.stack([
                anchor_boxes['1'][0, 0, 0:4], anchor_boxes['1'][0, 0, 8:12],
                anchor_boxes['1'][0, 1, 0:4], anchor_boxes['1'][1, 0, 8:12],
                anchor_boxes['2'][0, 0, 4:8]
            ]),
            # Where the second is + some as last
            tf.stack([
                anchor_boxes['1'][0, 1, 8:12], anchor_boxes['1'][1, 0, 0:4],
                anchor_boxes['1'][1, 1, 0:4], anchor_boxes['1'][1, 1, 8:12],
                anchor_boxes['2'][0, 0, 4:8]
            ]),
        ]))

  def test_decode_multilevel_with_tflite_nms(self):
    config = common.TFLitePostProcessingConfig().as_dict()
    generator = detection_generator.MultilevelDetectionGenerator(
        apply_nms=True,
        nms_version='tflite',
        box_coder_weights=[9, 8, 7, 6],
        tflite_post_processing_config=config,
    )
    raw_scores = {
        '4': tf.zeros(shape=[1, 8, 8, 3 * 2], dtype=tf.float32),
        '5': tf.zeros(shape=[1, 4, 4, 3 * 2], dtype=tf.float32),
    }
    raw_boxes = {
        '4': tf.zeros(shape=[1, 8, 8, 4 * 2], dtype=tf.float32),
        '5': tf.zeros(shape=[1, 4, 4, 4 * 2], dtype=tf.float32),
    }
    anchor_boxes = {
        '4': tf.zeros(shape=[1, 8, 8, 4 * 2], dtype=tf.float32),
        '5': tf.zeros(shape=[1, 4, 4, 4 * 2], dtype=tf.float32),
    }

    expected_signature = (
        'name: "TFLite_Detection_PostProcess" attr { key: "max_detections"'
        ' value { i: 200 } } attr { key: "max_classes_per_detection" value { i:'
        ' 5 } } attr { key: "detections_per_class" value { i: 5 } } attr { key:'
        ' "use_regular_nms" value { b: false } } attr { key:'
        ' "nms_score_threshold" value { f: 0.100000 } } attr { key:'
        ' "nms_iou_threshold" value { f: 0.500000 } } attr { key: "y_scale"'
        ' value { f: 9.000000 } } attr { key: "x_scale" value { f: 8.000000 } }'
        ' attr { key: "h_scale" value { f: 7.000000 } } attr { key: "w_scale"'
        ' value { f: 6.000000 } } attr { key: "num_classes" value { i: 3 } }'
    )

    with mock.patch.object(
        tf, 'function', wraps=tf.function
    ) as mock_tf_function:
      test_output = generator(
          raw_boxes=raw_boxes,
          raw_scores=raw_scores,
          anchor_boxes=anchor_boxes,
          image_shape=tf.constant([], dtype=tf.int32),
      )
      mock_tf_function.assert_called_once_with(
          experimental_implements=expected_signature
      )

    self.assertEqual(
        test_output['num_detections'], tf.constant(0.0, dtype=tf.float32)
    )
    self.assertEqual(
        test_output['detection_boxes'], tf.constant(0.0, dtype=tf.float32)
    )
    self.assertEqual(
        test_output['detection_classes'], tf.constant(0.0, dtype=tf.float32)
    )
    self.assertEqual(
        test_output['detection_scores'], tf.constant(0.0, dtype=tf.float32)
    )

  def test_decode_multilevel_tflite_nms_error_on_wrong_boxes_shape(self):
    config = common.TFLitePostProcessingConfig().as_dict()
    generator = detection_generator.MultilevelDetectionGenerator(
        apply_nms=True,
        nms_version='tflite',
        tflite_post_processing_config=config,
    )
    raw_scores = {'4': tf.zeros(shape=[1, 4, 4, 3 * 2], dtype=tf.float32)}
    raw_boxes = {'4': tf.zeros(shape=[1, 4, 4, 3], dtype=tf.float32)}
    anchor_boxes = {'4': tf.zeros(shape=[1, 4, 4, 4 * 2], dtype=tf.float32)}
    with self.assertRaisesRegex(
        ValueError,
        'The last dimension of predicted boxes should be divisible by 4.',
    ):
      generator(
          raw_boxes=raw_boxes,
          raw_scores=raw_scores,
          anchor_boxes=anchor_boxes,
          image_shape=tf.constant([], dtype=tf.int32),
      )

  def test_decode_multilevel_tflite_nms_error_on_wrong_scores_shape(self):
    config = common.TFLitePostProcessingConfig().as_dict()
    generator = detection_generator.MultilevelDetectionGenerator(
        apply_nms=True,
        nms_version='tflite',
        tflite_post_processing_config=config,
    )
    raw_scores = {'4': tf.zeros(shape=[1, 4, 4, 7 * 3], dtype=tf.float32)}
    raw_boxes = {'4': tf.zeros(shape=[1, 4, 4, 4 * 5], dtype=tf.float32)}
    anchor_boxes = {'4': tf.zeros(shape=[1, 4, 4, 4 * 5], dtype=tf.float32)}
    with self.assertRaisesRegex(
        ValueError,
        'The last dimension of predicted scores should be divisible by',
    ):
      generator(
          raw_boxes=raw_boxes,
          raw_scores=raw_scores,
          anchor_boxes=anchor_boxes,
          image_shape=tf.constant([], dtype=tf.int32),
      )

  def test_serialize_deserialize(self):
    tflite_post_processing_config = {
        'max_detections': 100,
        'max_classes_per_detection': 1,
        'use_regular_nms': True,
        'nms_score_threshold': 0.01,
        'nms_iou_threshold': 0.5,
        'input_image_size': [224, 224],
    }
    kwargs = {
        'apply_nms': True,
        'pre_nms_top_k': 1000,
        'pre_nms_score_threshold': 0.1,
        'nms_iou_threshold': 0.5,
        'max_num_detections': 10,
        'nms_version': 'v2',
        'use_cpu_nms': False,
        'soft_nms_sigma': None,
        'tflite_post_processing_config': tflite_post_processing_config,
        'return_decoded': False,
        'use_class_agnostic_nms': False,
        'box_coder_weights': None,
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
