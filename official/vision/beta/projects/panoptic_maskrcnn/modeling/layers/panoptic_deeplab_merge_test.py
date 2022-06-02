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

"""Test for panoptic_deeplab_merge.py.

Note that the tests are branched from
https://raw.githubusercontent.com/google-research/deeplab2/main/model/post_processor/panoptic_deeplab_test.py
"""
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.panoptic_maskrcnn.modeling.layers import panoptic_deeplab_merge


class PostProcessingTest(tf.test.TestCase):

  def test_py_func_merge_semantic_and_instance_maps_can_run(self):
    batch = 1
    height = 5
    width = 5
    semantic_prediction = tf.random.uniform((batch, height, width),
                                            minval=0,
                                            maxval=20,
                                            dtype=tf.int32)
    instance_maps = tf.random.uniform((batch, height, width),
                                      minval=0,
                                      maxval=3,
                                      dtype=tf.int32)
    thing_class_ids = tf.convert_to_tensor([1, 2, 3])
    label_divisor = 256
    stuff_area_limit = 3
    void_label = 255
    panoptic_prediction = panoptic_deeplab_merge._merge_semantic_and_instance_maps(
        semantic_prediction, instance_maps, thing_class_ids, label_divisor,
        stuff_area_limit, void_label)
    self.assertListEqual(semantic_prediction.get_shape().as_list(),
                         panoptic_prediction.get_shape().as_list())

  def test_merge_semantic_and_instance_maps_with_a_simple_example(self):
    semantic_prediction = tf.convert_to_tensor(
        [[[0, 0, 0, 0],
          [0, 1, 1, 0],
          [0, 2, 2, 0],
          [2, 2, 3, 3]]], dtype=tf.int32)
    instance_maps = tf.convert_to_tensor(
        [[[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 1, 1, 0],
          [2, 2, 3, 3]]], dtype=tf.int32)
    thing_class_ids = tf.convert_to_tensor([2, 3])
    label_divisor = 256
    stuff_area_limit = 3
    void_label = 255
    # The expected_panoptic_prediction is computed as follows.
    # For `thing` segmentation, instance 1, 2, and 3 are kept, but instance 3
    # will have a new instance ID 1, since it is the first instance in its
    # own semantic label.
    # For `stuff` segmentation, class-0 region is kept, while class-1 region
    # is re-labeled as `void_label * label_divisor` since its area is smaller
    # than stuff_area_limit.
    expected_panoptic_prediction = tf.convert_to_tensor(
        [[[0, 0, 0, 0],
          [0, void_label * label_divisor, void_label * label_divisor, 0],
          [0, 2 * label_divisor + 1, 2 * label_divisor + 1, 0],
          [2 * label_divisor + 2, 2 * label_divisor + 2, 3 * label_divisor + 1,
           3 * label_divisor + 1]]], dtype=tf.int32)
    panoptic_prediction = panoptic_deeplab_merge._merge_semantic_and_instance_maps(
        semantic_prediction, instance_maps, thing_class_ids, label_divisor,
        stuff_area_limit, void_label)
    self.assertAllClose(expected_panoptic_prediction,
                        panoptic_prediction)

  def test_gets_panoptic_predictions_with_score(self):
    batch = 1
    height = 5
    width = 5
    classes = 3

    semantic_logits = tf.random.uniform((batch, 1, 1, classes))
    semantic_logits = tf.tile(semantic_logits, (1, height, width, 1))

    center_heatmap = tf.convert_to_tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.7],
        [0.0, 0.0, 0.0, 0.0, 0.2],
    ], dtype=tf.float32)
    center_heatmap = tf.expand_dims(center_heatmap, 0)
    center_heatmap = tf.expand_dims(center_heatmap, 3)

    center_offsets = tf.zeros((batch, height, width, 2))
    center_threshold = 0.0
    thing_class_ids = tf.range(classes)  # No "stuff" classes.
    label_divisor = 256
    stuff_area_limit = 16
    void_label = classes
    nms_kernel_size = 3
    keep_k_centers = 2

    result = panoptic_deeplab_merge._get_panoptic_predictions(
        semantic_logits, center_heatmap, center_offsets, center_threshold,
        thing_class_ids, label_divisor, stuff_area_limit, void_label,
        nms_kernel_size, keep_k_centers)
    instance_maps = result[3].numpy()
    instance_scores = result[2].numpy()

    self.assertSequenceEqual(instance_maps.shape, (batch, height, width))
    expected_instances = [[
        [1, 1, 1, 1, 2],
        [1, 1, 1, 2, 2],
        [1, 1, 2, 2, 2],
        [1, 2, 2, 2, 2],
        [1, 2, 2, 2, 2],
    ]]
    np.testing.assert_array_equal(instance_maps, expected_instances)

    self.assertSequenceEqual(instance_scores.shape, (batch, height, width))
    expected_instance_scores = [[
        [1.0, 1.0, 1.0, 1.0, 0.7],
        [1.0, 1.0, 1.0, 0.7, 0.7],
        [1.0, 1.0, 0.7, 0.7, 0.7],
        [1.0, 0.7, 0.7, 0.7, 0.7],
        [1.0, 0.7, 0.7, 0.7, 0.7],
    ]]
    self.assertAllClose(result[2],
                        tf.constant(expected_instance_scores))


if __name__ == '__main__':
  tf.test.main()
