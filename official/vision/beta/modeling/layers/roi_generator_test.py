# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for roi_generator.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.layers import roi_generator


class MultilevelProposeRoisTest(tf.test.TestCase):

  def test_multilevel_propose_rois_single_level(self):
    rpn_boxes_np = np.array(
        [[[[0, 0, 10, 10], [0.01, 0.01, 9.9, 9.9]],
          [[5, 5, 10, 10], [2, 2, 8, 8]]],
         [[[2, 2, 4, 4], [3, 3, 6, 6]],
          [[3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]]]])
    rpn_boxes = {
        2: tf.constant(rpn_boxes_np, dtype=tf.float32)
    }
    rpn_scores_np = np.array(
        [[[[0.6], [0.9]], [[0.2], [0.3]]], [[[0.1], [0.8]], [[0.3], [0.5]]]])
    rpn_scores = {
        2: tf.constant(rpn_scores_np, dtype=tf.float32)
    }
    anchor_boxes_np = np.array(
        [[[[0, 0, 10, 10], [0.01, 0.01, 9.9, 9.9]],
          [[5, 5, 10, 10], [2, 2, 8, 8]]],
         [[[2, 2, 4, 4], [3, 3, 6, 6]],
          [[3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]]]])
    anchor_boxes = {
        2: tf.constant(anchor_boxes_np, dtype=tf.float32)
    }
    image_shape = tf.constant([[20, 20], [20, 20]], dtype=tf.int32)

    selected_rois_np = np.array(
        [[[0.01, 0.01, 9.9, 9.9], [2, 2, 8, 8], [5, 5, 10, 10], [0, 0, 0, 0]],
         [[3, 3, 6, 6], [1, 1, 8, 8], [2, 2, 4, 4], [0, 0, 0, 0]]])
    selected_roi_scores_np = np.array(
        [[0.9, 0.3, 0.2, 0], [0.8, 0.5, 0.1, 0]])

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      selected_rois_tpu, selected_roi_scores_tpu = (
          roi_generator._multilevel_propose_rois(
              rpn_boxes,
              rpn_scores,
              anchor_boxes=anchor_boxes,
              image_shape=image_shape,
              pre_nms_top_k=4,
              pre_nms_score_threshold=0.0,
              pre_nms_min_size_threshold=0.0,
              nms_iou_threshold=0.5,
              num_proposals=4,
              use_batched_nms=False,
              decode_boxes=False,
              clip_boxes=False,
              apply_sigmoid_to_score=False))

    # Runs on CPU.
    selected_rois_cpu, selected_roi_scores_cpu = (
        roi_generator._multilevel_propose_rois(
            rpn_boxes,
            rpn_scores,
            anchor_boxes=anchor_boxes,
            image_shape=image_shape,
            pre_nms_top_k=4,
            pre_nms_score_threshold=0.0,
            pre_nms_min_size_threshold=0.0,
            nms_iou_threshold=0.5,
            num_proposals=4,
            use_batched_nms=False,
            decode_boxes=False,
            clip_boxes=False,
            apply_sigmoid_to_score=False))

    self.assertNDArrayNear(
        selected_rois_tpu.numpy(), selected_rois_cpu.numpy(), 1e-5)
    self.assertNDArrayNear(
        selected_roi_scores_tpu.numpy(), selected_roi_scores_cpu.numpy(), 1e-5)

    self.assertNDArrayNear(
        selected_rois_tpu.numpy(), selected_rois_np, 1e-5)
    self.assertNDArrayNear(
        selected_roi_scores_tpu.numpy(), selected_roi_scores_np, 1e-5)

  def test_multilevel_propose_rois_two_levels(self):
    rpn_boxes_1_np = np.array(
        [[[[0, 0, 10, 10], [0.01, 0.01, 9.99, 9.99]],
          [[5, 5, 10, 10], [2, 2, 8, 8]]],
         [[[2, 2, 2.5, 2.5], [3, 3, 6, 6]],
          [[3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]]]])
    rpn_boxes_2_np = np.array(
        [[[[0, 0, 10.01, 10.01]]], [[[2, 2, 4.5, 4.5]]]])
    rpn_boxes = {
        2: tf.constant(rpn_boxes_1_np, dtype=tf.float32),
        3: tf.constant(rpn_boxes_2_np, dtype=tf.float32),
    }
    rpn_scores_1_np = np.array(
        [[[[0.6], [0.9]], [[0.2], [0.3]]], [[[0.1], [0.8]], [[0.3], [0.5]]]])
    rpn_scores_2_np = np.array([[[[0.95]]], [[[0.99]]]])
    rpn_scores = {
        2: tf.constant(rpn_scores_1_np, dtype=tf.float32),
        3: tf.constant(rpn_scores_2_np, dtype=tf.float32),
    }
    anchor_boxes_1_np = np.array(
        [[[[0, 0, 10, 10], [0.01, 0.01, 9.99, 9.99]],
          [[5, 5, 10, 10], [2, 2, 8, 8]]],
         [[[2, 2, 2.5, 2.5], [3, 3, 6, 6]],
          [[3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]]]])
    anchor_boxes_2_np = np.array(
        [[[[0, 0, 10.01, 10.01]]], [[[2, 2, 4.5, 4.5]]]])
    anchor_boxes = {
        2: tf.constant(anchor_boxes_1_np, dtype=tf.float32),
        3: tf.constant(anchor_boxes_2_np, dtype=tf.float32),
    }
    image_shape = tf.constant([[20, 20], [20, 20]], dtype=tf.int32)

    selected_rois_np = np.array(
        [[[0, 0, 10.01, 10.01], [0.01, 0.01, 9.99, 9.99]],
         [[2, 2, 4.5, 4.5], [3, 3, 6, 6]]])
    selected_roi_scores_np = np.array([[0.95, 0.9], [0.99, 0.8]])

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      selected_rois_tpu, selected_roi_scores_tpu = (
          roi_generator._multilevel_propose_rois(
              rpn_boxes,
              rpn_scores,
              anchor_boxes=anchor_boxes,
              image_shape=image_shape,
              pre_nms_top_k=4,
              pre_nms_score_threshold=0.0,
              pre_nms_min_size_threshold=0.0,
              nms_iou_threshold=0.5,
              num_proposals=2,
              use_batched_nms=False,
              decode_boxes=False,
              clip_boxes=False,
              apply_sigmoid_to_score=False))

    # Runs on CPU.
    selected_rois_cpu, selected_roi_scores_cpu = (
        roi_generator._multilevel_propose_rois(
            rpn_boxes,
            rpn_scores,
            anchor_boxes=anchor_boxes,
            image_shape=image_shape,
            pre_nms_top_k=4,
            pre_nms_score_threshold=0.0,
            pre_nms_min_size_threshold=0.0,
            nms_iou_threshold=0.5,
            num_proposals=2,
            use_batched_nms=False,
            decode_boxes=False,
            clip_boxes=False,
            apply_sigmoid_to_score=False))

    self.assertNDArrayNear(
        selected_rois_tpu.numpy(), selected_rois_cpu.numpy(), 1e-5)
    self.assertNDArrayNear(
        selected_roi_scores_tpu.numpy(), selected_roi_scores_cpu.numpy(), 1e-5)

    self.assertNDArrayNear(
        selected_rois_tpu.numpy(), selected_rois_np, 1e-5)
    self.assertNDArrayNear(
        selected_roi_scores_tpu.numpy(), selected_roi_scores_np, 1e-5)


class MultilevelROIGeneratorTest(tf.test.TestCase):

  def test_serialize_deserialize(self):
    kwargs = dict(
        pre_nms_top_k=2000,
        pre_nms_score_threshold=0.0,
        pre_nms_min_size_threshold=0.0,
        nms_iou_threshold=0.7,
        num_proposals=1000,
        test_pre_nms_top_k=1000,
        test_pre_nms_score_threshold=0.0,
        test_pre_nms_min_size_threshold=0.0,
        test_nms_iou_threshold=0.7,
        test_num_proposals=1000,
        use_batched_nms=False,
    )
    generator = roi_generator.MultilevelROIGenerator(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(generator.get_config(), expected_config)

    new_generator = roi_generator.MultilevelROIGenerator.from_config(
        generator.get_config())

    self.assertAllEqual(generator.get_config(), new_generator.get_config())
