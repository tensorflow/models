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

"""Tests for panoptic_segmentation_generator.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations

from official.vision.beta.projects.panoptic_maskrcnn.modeling.layers import panoptic_segmentation_generator

PANOPTIC_SEGMENTATION_GENERATOR = panoptic_segmentation_generator.PanopticSegmentationGenerator


class PanopticSegmentationGeneratorTest(
    parameterized.TestCase, tf.test.TestCase):

  def test_serialize_deserialize(self):
    config = {
        'output_size': [640, 640],
        'max_num_detections': 100,
        'stuff_classes_offset': 90,
        'mask_binarize_threshold': 0.5,
        'score_threshold': 0.005,
        'things_class_label': 1,
        'void_class_label': 0,
        'void_instance_id': -1,
        'rescale_predictions': False,
    }
    generator = PANOPTIC_SEGMENTATION_GENERATOR(**config)

    expected_config = dict(config)
    self.assertEqual(generator.get_config(), expected_config)

    new_generator = PANOPTIC_SEGMENTATION_GENERATOR.from_config(
        generator.get_config())

    self.assertAllEqual(generator.get_config(), new_generator.get_config())

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.default_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ]))
  def test_outputs(self, strategy):

    # 0 represents the void class label
    thing_class_ids = [0, 1, 2, 3, 4]
    stuff_class_ids = [0, 5, 6, 7, 8, 9, 10]
    all_class_ids = set(thing_class_ids + stuff_class_ids)

    num_thing_classes = len(thing_class_ids)
    num_stuff_classes = len(stuff_class_ids)
    num_classes_for_segmentation = num_stuff_classes + 1

    # all thing classes are mapped to class_id=1, stuff class ids are offset
    # such that the stuff class_ids start from 2, this means the semantic
    # segmentation head will have ground truths with class_ids belonging to
    # [0, 1, 2, 3, 4, 5, 6, 7]

    config = {
        'output_size': [640, 640],
        'max_num_detections': 100,
        'stuff_classes_offset': 3,
        'mask_binarize_threshold': 0.5,
        'score_threshold': 0.005,
        'things_class_label': 1,
        'void_class_label': 0,
        'void_instance_id': -1,
        'rescale_predictions': False,
    }
    generator = PANOPTIC_SEGMENTATION_GENERATOR(**config)

    crop_height = 112
    crop_width = 112

    boxes = tf.constant([[
        [167, 398, 342, 619],
        [192, 171, 363, 449],
        [211, 1, 382, 74]
    ]])

    num_detections = boxes.get_shape().as_list()[1]
    scores = tf.random.uniform([1, num_detections], 0, 1)
    classes = tf.random.uniform(
        [1, num_detections],
        1, num_thing_classes, dtype=tf.int32)
    masks = tf.random.normal(
        [1, num_detections, crop_height, crop_width])

    segmentation_mask = tf.random.uniform(
        [1, *config['output_size']],
        0, num_classes_for_segmentation, dtype=tf.int32)
    segmentation_mask_one_hot = tf.one_hot(
        segmentation_mask, depth=num_stuff_classes + 1)

    inputs = {
        'detection_boxes': boxes,
        'detection_scores': scores,
        'detection_classes': classes,
        'detection_masks': masks,
        'num_detections': tf.constant([num_detections]),
        'segmentation_outputs': segmentation_mask_one_hot
        }

    def _run(inputs):
      return generator(inputs=inputs)

    @tf.function
    def _distributed_run(inputs):
      outputs = strategy.run(_run, args=((inputs,)))
      return strategy.gather(outputs, axis=0)

    outputs = _distributed_run(inputs)

    self.assertIn('category_mask', outputs)
    self.assertIn('instance_mask', outputs)

    self.assertAllEqual(
        outputs['category_mask'][0].get_shape().as_list(),
        config['output_size'])

    self.assertAllEqual(
        outputs['instance_mask'][0].get_shape().as_list(),
        config['output_size'])

    for category_id in np.unique(outputs['category_mask']):
      self.assertIn(category_id, all_class_ids)

if __name__ == '__main__':
  tf.test.main()
