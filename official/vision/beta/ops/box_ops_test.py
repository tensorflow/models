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
"""Tests for box_ops.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.ops import box_ops


def _transform_boxes_on_tpu_and_cpu(transform_fn, boxes, *args):
  # Runs on TPU.
  strategy = tf.distribute.experimental.TPUStrategy()
  with strategy.scope():
    transformed_op_tpu = transform_fn(boxes, *args)
    transfomred_boxes_tpu = tf.nest.map_structure(lambda x: x.numpy(),
                                                  transformed_op_tpu)

  # Runs on CPU.
  transfomred_op_cpu = transform_fn(boxes, *args)
  transfomred_boxes_cpu = tf.nest.map_structure(lambda x: x.numpy(),
                                                transfomred_op_cpu)
  return transfomred_boxes_tpu, transfomred_boxes_cpu


class ConvertBoxesTest(tf.test.TestCase):

  def testConvertBoxes(self):
    # y1, x1, y2, x2.
    boxes = np.array([[0, 0, 1, 2], [0.2, 0.1, 1.2, 1.1]])
    # x1, y1, width, height
    target = np.array([[0, 0, 2, 1], [0.1, 0.2, 1, 1]])
    outboxes = box_ops.yxyx_to_xywh(boxes)
    self.assertNDArrayNear(outboxes, target, 1e-7)


class JitterBoxesTest(tf.test.TestCase):

  def testJitterBoxes(self):
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                  [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]]
    boxes_np = np.array(boxes_data, dtype=np.float32)
    max_size = max(
        np.amax(boxes_np[:, 3] - boxes_np[:, 1]),
        np.amax(boxes_np[:, 2] - boxes_np[:, 0]))
    noise_scale = 0.025
    boxes = tf.constant(boxes_np)

    def jitter_fn(input_boxes, arg_noise_scale):
      return box_ops.jitter_boxes(input_boxes, arg_noise_scale)

    jittered_boxes_tpu, jittered_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        jitter_fn, boxes, noise_scale)

    # Test that the jittered box is within 10 stds from the inputs.
    self.assertNDArrayNear(jittered_boxes_tpu, boxes_np,
                           noise_scale * max_size * 10)
    self.assertNDArrayNear(jittered_boxes_cpu, boxes_np,
                           noise_scale * max_size * 10)


class NormalizeBoxesTest(tf.test.TestCase):

  def testNormalizeBoxes1DWithImageShapeAsList(self):
    boxes = tf.constant([10, 30, 40, 90], tf.float32)
    image_shape = [50, 100]

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [0.2, 0.3, 0.8, 0.9], 1e-5)

  def testNormalizeBoxes1DWithImageShapeAsTensor(self):
    boxes = tf.constant([10, 30, 40, 90], tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [0.2, 0.3, 0.8, 0.9], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsList(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = [50, 100]

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsVector(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = tf.constant([50, 100], dtype=tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = tf.constant([[50, 100]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsSameShapeTensor(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = tf.constant([[50, 100], [50, 100]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsList(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = [50, 100]

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                            [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsVector(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                            [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = tf.constant([[[50, 100]], [[500, 1000]]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(
        normalized_boxes_tpu,
        [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
         [[0.04, 0.04, 0.1, 0.08], [0.06, 0.05, 0.08, 0.09]]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsSameShapeTensor(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = tf.constant(
        [[[50, 100], [50, 100]], [[500, 1000], [500, 1000]]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.normalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(
        normalized_boxes_tpu,
        [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
         [[0.04, 0.04, 0.1, 0.08], [0.06, 0.05, 0.08, 0.09]]], 1e-5)


class DenormalizeBoxesTest(tf.test.TestCase):

  def testDenormalizeBoxes1DWithImageShapeAsList(self):
    boxes = tf.constant([0.2, 0.3, 0.8, 0.9], tf.float32)
    image_shape = [50, 100]
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [10, 30, 40, 90], 1e-5)

  def testDenormalizeBoxes1DWithImageShapeAsTensor(self):
    boxes = tf.constant([0.2, 0.3, 0.8, 0.9], tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [10, 30, 40, 90], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsList(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = [50, 100]
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsVector(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = tf.constant([50, 100], dtype=tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = tf.constant([[50, 100]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsSameShapeTensor(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = tf.constant([[50, 100], [50, 100]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsList(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]],
                        tf.float32)
    image_shape = [50, 100]
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[10, 30, 40, 90], [30, 10, 40, 50]],
                            [[20, 40, 50, 80], [30, 50, 40, 90]]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsVector(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]],
                        tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[10, 30, 40, 90], [30, 10, 40, 50]],
                            [[20, 40, 50, 80], [30, 50, 40, 90]]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.04, 0.04, 0.1, 0.08], [0.06, 0.05, 0.08, 0.09]]],
                        tf.float32)
    image_shape = tf.constant([[[50, 100]], [[500, 1000]]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[10, 30, 40, 90], [30, 10, 40, 50]],
                            [[20, 40, 50, 80], [30, 50, 40, 90]]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsSameShapeTensor(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.04, 0.04, 0.1, 0.08], [0.06, 0.05, 0.08, 0.09]]],
                        tf.float32)
    image_shape = tf.constant(
        [[[50, 100], [50, 100]], [[500, 1000], [500, 1000]]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            box_ops.denormalize_boxes, boxes, image_shape))

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[10, 30, 40, 90], [30, 10, 40, 50]],
                            [[20, 40, 50, 80], [30, 50, 40, 90]]], 1e-5)


class ClipBoxesTest(tf.test.TestCase):

  def testClipBoxesImageShapeAsList(self):
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                  [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]]
    image_shape = [3, 3]
    boxes = tf.constant(boxes_data)

    clipped_boxes_tpu, clipped_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_ops.clip_boxes, boxes, image_shape)

    self.assertAllClose(clipped_boxes_tpu, clipped_boxes_cpu)
    self.assertAllClose(clipped_boxes_tpu,
                        [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                         [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]])

  def testClipBoxesImageShapeAsVector(self):
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                  [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]]
    boxes = tf.constant(boxes_data)
    image_shape = np.array([3, 3], dtype=np.float32)
    clipped_boxes_tpu, clipped_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_ops.clip_boxes, boxes, image_shape)

    self.assertAllClose(clipped_boxes_tpu, clipped_boxes_cpu)
    self.assertAllClose(clipped_boxes_tpu,
                        [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                         [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]])

  def testClipBoxesImageShapeAsTensor(self):
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                  [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]]
    boxes = tf.constant(boxes_data)
    image_shape = tf.constant([[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                              dtype=tf.float32)
    clipped_boxes_tpu, clipped_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_ops.clip_boxes, boxes, image_shape)

    self.assertAllClose(clipped_boxes_tpu, clipped_boxes_cpu)
    self.assertAllClose(clipped_boxes_tpu,
                        [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, 0.3, 1, 1.3],
                         [0, 0.5, 1, 1.5], [0, 0.7, 1, 1.7], [0, 1.9, 1, 1.9]])


class EncodeDecodeBoxesTest(tf.test.TestCase):

  def test_encode_decode_boxes(self):
    boxes_np = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
                         [[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    anchors = tf.constant([[[1.5, 2.5, 3.5, 4.5], [2.5, 3.5, 4.5, 5.5]],
                           [[1.5, 2.5, 3.5, 4.5], [2.5, 3.5, 4.5, 5.5]]],
                          dtype=tf.float32)
    weights = [1.0, 1.0, 1.0, 1.0]

    def test_fn(boxes, anchors):
      encoded_boxes = box_ops.encode_boxes(boxes, anchors, weights)
      decoded_boxes = box_ops.decode_boxes(encoded_boxes, anchors, weights)
      return decoded_boxes

    decoded_boxes_tpu, decoded_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        test_fn, boxes, anchors)

    self.assertNDArrayNear(decoded_boxes_tpu, decoded_boxes_cpu, 1e-5)
    self.assertNDArrayNear(decoded_boxes_tpu, boxes_np, 1e-5)

  def test_encode_decode_boxes_batch_broadcast(self):
    boxes_np = np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
                         [[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    anchors = tf.constant([[[1.5, 2.5, 3.5, 4.5], [2.5, 3.5, 4.5, 5.5]]],
                          dtype=tf.float32)
    weights = [1.0, 1.0, 1.0, 1.0]

    def test_fn(boxes, anchors):
      encoded_boxes = box_ops.encode_boxes(boxes, anchors, weights)
      decoded_boxes = box_ops.decode_boxes(encoded_boxes, anchors, weights)
      return decoded_boxes

    decoded_boxes_tpu, decoded_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        test_fn, boxes, anchors)

    self.assertNDArrayNear(decoded_boxes_tpu, decoded_boxes_cpu, 1e-5)
    self.assertNDArrayNear(decoded_boxes_tpu, boxes_np, 1e-5)


class FilterBoxesTest(tf.test.TestCase):

  def test_filter_boxes_batch(self):
    # boxes -> [[small, good, outside], [outside, small, good]]
    boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5],
                          [7.0, 4.0, 9.5, 6.5]],
                         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0],
                          [4.0, 1.0, 7.0, 4.0]]])
    filtered_boxes_np = np.array([[[0.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.5, 5.5],
                                   [0.0, 0.0, 0.0, 0.0]],
                                  [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                   [4.0, 1.0, 7.0, 4.0]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    scores_np = np.array([[0.9, 0.7, 0.5], [0.11, 0.22, 0.33]])
    filtered_scores_np = np.array([[0.0, 0.7, 0.0], [0.0, 0.0, 0.33]])
    scores = tf.constant(scores_np, dtype=tf.float32)
    image_shape = tf.expand_dims(
        tf.constant([[8, 8], [8, 8]], dtype=tf.int32), axis=1)
    min_size_threshold = 2.0

    def test_fn(boxes, scores, image_shape):
      filtered_boxes, filtered_scores = box_ops.filter_boxes(
          boxes, scores, image_shape, min_size_threshold)
      return filtered_boxes, filtered_scores

    filtered_results_tpu, filtered_results_cpu = (
        _transform_boxes_on_tpu_and_cpu(
            test_fn, boxes, scores, image_shape))
    filtered_boxes_tpu, filtered_scores_tpu = filtered_results_tpu
    filtered_boxes_cpu, filtered_scores_cpu = filtered_results_cpu

    self.assertNDArrayNear(filtered_boxes_tpu, filtered_boxes_cpu, 1e-5)
    self.assertNDArrayNear(filtered_scores_tpu, filtered_scores_cpu, 1e-5)
    self.assertNDArrayNear(filtered_boxes_tpu, filtered_boxes_np, 1e-5)
    self.assertNDArrayNear(filtered_scores_tpu, filtered_scores_np, 1e-5)


class FilterBoxesByScoresTest(tf.test.TestCase):

  def test_filter_boxes_by_scores_batch(self):
    # boxes -> [[small, good, outside], [outside, small, good]]
    boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5],
                          [7.0, 4.0, 9.5, 6.5]],
                         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0],
                          [4.0, 1.0, 7.0, 4.0]]])
    filtered_boxes_np = np.array([[[0.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.5, 5.5],
                                   [7.0, 4.0, 9.5, 6.5]],
                                  [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                                   [4.0, 1.0, 7.0, 4.0]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    scores_np = np.array([[0.1, 0.7, 0.6], [0.11, 0.22, 0.53]])
    filtered_scores_np = np.array([[-1.0, 0.7, 0.6], [-1.0, -1.0, 0.53]])
    scores = tf.constant(scores_np, dtype=tf.float32)
    min_score_threshold = 0.5

    def test_fn(boxes, scores):
      filtered_boxes, filtered_scores = box_ops.filter_boxes_by_scores(
          boxes, scores, min_score_threshold)
      return filtered_boxes, filtered_scores

    filtered_results_tpu, filtered_results_cpu = _transform_boxes_on_tpu_and_cpu(
        test_fn, boxes, scores)
    filtered_boxes_tpu, filtered_scores_tpu = filtered_results_tpu
    filtered_boxes_cpu, filtered_scores_cpu = filtered_results_cpu

    self.assertNDArrayNear(filtered_boxes_tpu, filtered_boxes_cpu, 1e-5)
    self.assertNDArrayNear(filtered_scores_tpu, filtered_scores_cpu, 1e-5)
    self.assertNDArrayNear(filtered_boxes_tpu, filtered_boxes_np, 1e-5)
    self.assertNDArrayNear(filtered_scores_tpu, filtered_scores_np, 1e-5)


class GatherInstancesTest(tf.test.TestCase):

  def test_gather_instances(self):
    boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5],
                          [7.0, 4.0, 9.5, 6.5]],
                         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0],
                          [4.0, 1.0, 7.0, 4.0]]])
    indices_np = np.array([[2, 0], [0, 1]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    indices = tf.constant(indices_np, dtype=tf.int32)

    selected_boxes = box_ops.gather_instances(indices, boxes)

    expected_selected_boxes = np.array(
        [[[7.0, 4.0, 9.5, 6.5], [1.0, 2.0, 1.5, 2.5]],
         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0]]])

    self.assertNDArrayNear(expected_selected_boxes, selected_boxes, 1e-5)

  def test_gather_instances_with_multiple_inputs(self):
    boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5],
                          [7.0, 4.0, 9.5, 6.5]],
                         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0],
                          [4.0, 1.0, 7.0, 4.0]]])
    classes_np = np.array([[1, 2, 3], [20, 30, 40]])
    indices_np = np.array([[2, 0], [0, 1]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    classes = tf.constant(classes_np, dtype=tf.int32)
    indices = tf.constant(indices_np, dtype=tf.int32)

    selected_boxes, selected_classes = box_ops.gather_instances(
        indices, boxes, classes)

    expected_selected_boxes = np.array(
        [[[7.0, 4.0, 9.5, 6.5], [1.0, 2.0, 1.5, 2.5]],
         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0]]])
    expected_selected_classes = np.array(
        [[3, 1], [20, 30]])

    self.assertNDArrayNear(expected_selected_boxes, selected_boxes, 1e-5)
    self.assertAllEqual(expected_selected_classes, selected_classes)


class TopKBoxesTest(tf.test.TestCase):

  def test_top_k_boxes_batch1(self):
    boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5],
                          [7.0, 4.0, 9.5, 6.5]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    scores_np = np.array([[0.9, 0.5, 0.7]])
    scores = tf.constant(scores_np, dtype=tf.float32)
    top_k_boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [7.0, 4.0, 9.5, 6.5]]])
    top_k_scores_np = np.array([[0.9, 0.7]])

    def test_fn(boxes, scores):
      top_k_boxes, top_k_scores = box_ops.top_k_boxes(boxes, scores, k=2)
      return top_k_boxes, top_k_scores

    top_k_results_tpu, top_k_results_cpu = _transform_boxes_on_tpu_and_cpu(
        test_fn, boxes, scores)
    top_k_boxes_tpu, top_k_scores_tpu = top_k_results_tpu
    top_k_boxes_cpu, top_k_scores_cpu = top_k_results_cpu

    self.assertNDArrayNear(top_k_boxes_tpu, top_k_boxes_cpu, 1e-5)
    self.assertNDArrayNear(top_k_scores_tpu, top_k_scores_cpu, 1e-5)
    self.assertNDArrayNear(top_k_boxes_tpu, top_k_boxes_np, 1e-5)
    self.assertNDArrayNear(top_k_scores_tpu, top_k_scores_np, 1e-5)

  def test_top_k_boxes_batch2(self):
    boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5],
                          [7.0, 4.0, 9.5, 6.5]],
                         [[-2.0, 5.0, 0.0, 7.5], [5.0, 6.0, 5.1, 6.0],
                          [4.0, 1.0, 7.0, 4.0]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)
    scores_np = np.array([[0.9, 0.7, 0.5], [0.11, 0.22, 0.33]])
    scores = tf.constant(scores_np, dtype=tf.float32)
    top_k_boxes_np = np.array([[[1.0, 2.0, 1.5, 2.5], [2.0, 3.0, 4.5, 5.5]],
                               [[4.0, 1.0, 7.0, 4.0], [5.0, 6.0, 5.1, 6.0]]])
    top_k_scores_np = np.array([[0.9, 0.7], [0.33, 0.22]])

    def test_fn(boxes, scores):
      top_k_boxes, top_k_scores = box_ops.top_k_boxes(boxes, scores, k=2)
      return top_k_boxes, top_k_scores

    top_k_results_tpu, top_k_results_cpu = _transform_boxes_on_tpu_and_cpu(
        test_fn, boxes, scores)
    top_k_boxes_tpu, top_k_scores_tpu = top_k_results_tpu
    top_k_boxes_cpu, top_k_scores_cpu = top_k_results_cpu

    self.assertNDArrayNear(top_k_boxes_tpu, top_k_boxes_cpu, 1e-5)
    self.assertNDArrayNear(top_k_scores_tpu, top_k_scores_cpu, 1e-5)
    self.assertNDArrayNear(top_k_boxes_tpu, top_k_boxes_np, 1e-5)
    self.assertNDArrayNear(top_k_scores_tpu, top_k_scores_np, 1e-5)


class BboxeOverlapTest(tf.test.TestCase):

  def testBBoxeOverlapOpCorrectness(self):
    boxes_data = [[[0, 0, 0.1, 1], [0, 0.2, 0.2, 1.2], [0, 0.3, 0.3, 1.3],
                   [0, 0.5, 0.4, 1.5], [0, 0.7, 0.5, 1.7], [0, 0.9, 0.6, 1.9],
                   [0, 0.1, 0.1, 1.1], [0, 0.3, 0.7, 1.3], [0, 0.9, 2, 1.9]],
                  [[0, 0, 1, 0.2], [0, 0.2, 0.5, 1.2], [0, 0.4, 0.9, 1.4],
                   [0, 0.6, 1.1, 1.6], [0, 0.8, 1.2, 1.8], [0, 1, 1.5, 2],
                   [0, 0.5, 1, 1], [0.5, 0.8, 1, 1.8], [-1, -1, -1, -1]]]
    boxes_np = np.array(boxes_data, dtype=np.float32)
    gt_boxes_data = [[[0, 0.1, 0.1, 1.1], [0, 0.3, 0.7, 1.3], [0, 0.9, 2, 1.9]],
                     [[0, 0.5, 1, 1], [0.5, 0.8, 1, 1.8], [-1, -1, -1, -1]]]
    gt_boxes_np = np.array(gt_boxes_data, dtype=np.float32)

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      boxes = tf.constant(boxes_np)
      gt_boxes = tf.constant(gt_boxes_np)
      iou = box_ops.bbox_overlap(boxes=boxes, gt_boxes=gt_boxes)
      iou = iou.numpy()
    self.assertEqual(iou.shape, (2, 9, 3))
    self.assertAllEqual(
        np.argmax(iou, axis=2),
        [[0, 0, 1, 1, 1, 2, 0, 1, 2], [0, 0, 0, 0, 1, 1, 0, 1, 0]])

  def testBBoxeOverlapOpCheckShape(self):
    batch_size = 2
    rpn_post_nms_topn = 2000
    gt_max_instances = 100
    boxes_np = np.random.rand(batch_size, rpn_post_nms_topn,
                              4).astype(np.float32)
    gt_boxes_np = np.random.rand(batch_size, gt_max_instances,
                                 4).astype(np.float32)
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      boxes = tf.constant(boxes_np)
      gt_boxes = tf.constant(gt_boxes_np)
      iou = box_ops.bbox_overlap(boxes=boxes, gt_boxes=gt_boxes)
      iou = iou.numpy()
    self.assertEqual(iou.shape,
                     (batch_size, (rpn_post_nms_topn), gt_max_instances))

  def testBBoxeOverlapOpCorrectnessWithNegativeData(self):
    boxes_data = [[[0, -0.01, 0.1, 1.1], [0, 0.2, 0.2, 5.0],
                   [0, -0.01, 0.1, 1.], [-1, -1, -1, -1]]]
    boxes_np = np.array(boxes_data, dtype=np.float32)
    gt_boxes_np = boxes_np
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      boxes = tf.constant(boxes_np)
      gt_boxes = tf.constant(gt_boxes_np)
      iou = box_ops.bbox_overlap(boxes=boxes, gt_boxes=gt_boxes)
      iou = iou.numpy()
    expected = np.array([[[0.99999994, 0.0917431, 0.9099099, -1.],
                          [0.0917431, 1., 0.08154944, -1.],
                          [0.9099099, 0.08154944, 1., -1.],
                          [-1., -1., -1., -1.]]])
    self.assertAllClose(expected, iou)


class BoxMatchingTest(tf.test.TestCase):

  def test_box_matching_single(self):
    boxes_np = np.array(
        [[[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5],
          [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)

    gt_boxes_np = np.array(
        [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5],
          [-1, -1, -1, -1]]])
    gt_boxes = tf.constant(gt_boxes_np, dtype=tf.float32)
    gt_classes_np = np.array([[2, 10, -1]])
    gt_classes = tf.constant(gt_classes_np, dtype=tf.int32)

    matched_gt_boxes_np = np.array(
        [[[2.5, 2.5, 7.5, 7.5],
          [2.5, 2.5, 7.5, 7.5],
          [2.5, 2.5, 7.5, 7.5],
          [10, 10, 15, 15]]])
    matched_gt_classes_np = np.array([[10, 10, 10, 2]])
    matched_gt_indices_np = np.array([[1, 1, 1, 0]])
    matched_iou_np = np.array(
        [[0.142857142857, 1.0, 0.142857142857, 0.142857142857]])
    iou_np = np.array(
        [[[0, 0.142857142857, -1.0],
          [0, 1.0, -1.0],
          [0, 0.142857142857, -1.0],
          [0.142857142857, 0, -1.0]]])

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      (matched_gt_boxes_tpu, matched_gt_classes_tpu,
       matched_gt_indices_tpu, matched_iou_tpu, iou_tpu) = (
           box_ops.box_matching(boxes, gt_boxes, gt_classes))

    # Runs on CPU.
    (matched_gt_boxes_cpu, matched_gt_classes_cpu,
     matched_gt_indices_cpu, matched_iou_cpu, iou_cpu) = (
         box_ops.box_matching(boxes, gt_boxes, gt_classes))

    # consistency.
    self.assertNDArrayNear(
        matched_gt_boxes_tpu.numpy(), matched_gt_boxes_cpu.numpy(), 1e-5)
    self.assertAllEqual(
        matched_gt_classes_tpu.numpy(), matched_gt_classes_cpu.numpy())
    self.assertAllEqual(
        matched_gt_indices_tpu.numpy(), matched_gt_indices_cpu.numpy())
    self.assertNDArrayNear(
        matched_iou_tpu.numpy(), matched_iou_cpu.numpy(), 1e-5)
    self.assertNDArrayNear(
        iou_tpu.numpy(), iou_cpu.numpy(), 1e-5)

    # correctness.
    self.assertNDArrayNear(
        matched_gt_boxes_tpu.numpy(), matched_gt_boxes_np, 1e-5)
    self.assertAllEqual(
        matched_gt_classes_tpu.numpy(), matched_gt_classes_np)
    self.assertAllEqual(
        matched_gt_indices_tpu.numpy(), matched_gt_indices_np)
    self.assertNDArrayNear(
        matched_iou_tpu.numpy(), matched_iou_np, 1e-5)
    self.assertNDArrayNear(
        iou_tpu.numpy(), iou_np, 1e-5)

  def test_box_matching_single_no_gt(self):
    boxes_np = np.array(
        [[[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5],
          [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)

    gt_boxes_np = np.array(
        [[[-1, -1, -1, -1],
          [-1, -1, -1, -1],
          [-1, -1, -1, -1]]])
    gt_boxes = tf.constant(gt_boxes_np, dtype=tf.float32)
    gt_classes_np = np.array([[-1, -1, -1]])
    gt_classes = tf.constant(gt_classes_np, dtype=tf.int32)

    matched_gt_boxes_np = np.array(
        [[[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]])
    matched_gt_classes_np = np.array([[0, 0, 0, 0]])
    matched_gt_indices_np = np.array([[-1, -1, -1, -1]])
    matched_iou_np = np.array([[-1, -1, -1, -1]])
    iou_np = np.array(
        [[[-1, -1, -1],
          [-1, -1, -1],
          [-1, -1, -1],
          [-1, -1, -1]]])

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      (matched_gt_boxes_tpu, matched_gt_classes_tpu,
       matched_gt_indices_tpu, matched_iou_tpu, iou_tpu) = (
           box_ops.box_matching(boxes, gt_boxes, gt_classes))

    # Runs on CPU.
    (matched_gt_boxes_cpu, matched_gt_classes_cpu,
     matched_gt_indices_cpu, matched_iou_cpu, iou_cpu) = (
         box_ops.box_matching(boxes, gt_boxes, gt_classes))

    # consistency.
    self.assertNDArrayNear(
        matched_gt_boxes_tpu.numpy(), matched_gt_boxes_cpu.numpy(), 1e-5)
    self.assertAllEqual(
        matched_gt_classes_tpu.numpy(), matched_gt_classes_cpu.numpy())
    self.assertAllEqual(
        matched_gt_indices_tpu.numpy(), matched_gt_indices_cpu.numpy())
    self.assertNDArrayNear(
        matched_iou_tpu.numpy(), matched_iou_cpu.numpy(), 1e-5)
    self.assertNDArrayNear(
        iou_tpu.numpy(), iou_cpu.numpy(), 1e-5)

    # correctness.
    self.assertNDArrayNear(
        matched_gt_boxes_tpu.numpy(), matched_gt_boxes_np, 1e-5)
    self.assertAllEqual(
        matched_gt_classes_tpu.numpy(), matched_gt_classes_np)
    self.assertAllEqual(
        matched_gt_indices_tpu.numpy(), matched_gt_indices_np)
    self.assertNDArrayNear(
        matched_iou_tpu.numpy(), matched_iou_np, 1e-5)
    self.assertNDArrayNear(
        iou_tpu.numpy(), iou_np, 1e-5)

  def test_box_matching_batch(self):
    boxes_np = np.array(
        [[[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5],
          [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]],
         [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5],
          [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]])
    boxes = tf.constant(boxes_np, dtype=tf.float32)

    gt_boxes_np = np.array(
        [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
         [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]]])
    gt_boxes = tf.constant(gt_boxes_np, dtype=tf.float32)
    gt_classes_np = np.array([[2, 10, -1], [-1, -1, -1]])
    gt_classes = tf.constant(gt_classes_np, dtype=tf.int32)

    matched_gt_boxes_np = np.array(
        [[[2.5, 2.5, 7.5, 7.5],
          [2.5, 2.5, 7.5, 7.5],
          [2.5, 2.5, 7.5, 7.5],
          [10, 10, 15, 15]],
         [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]])
    matched_gt_classes_np = np.array(
        [[10, 10, 10, 2],
         [0, 0, 0, 0]])
    matched_gt_indices_np = np.array(
        [[1, 1, 1, 0],
         [-1, -1, -1, -1]])
    matched_iou_np = np.array(
        [[0.142857142857, 1.0, 0.142857142857, 0.142857142857],
         [-1, -1, -1, -1]])
    iou_np = np.array(
        [[[0, 0.142857142857, -1.0],
          [0, 1.0, -1.0],
          [0, 0.142857142857, -1.0],
          [0.142857142857, 0, -1.0]],
         [[-1, -1, -1],
          [-1, -1, -1],
          [-1, -1, -1],
          [-1, -1, -1]]])

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      (matched_gt_boxes_tpu, matched_gt_classes_tpu,
       matched_gt_indices_tpu, matched_iou_tpu, iou_tpu) = (
           box_ops.box_matching(boxes, gt_boxes, gt_classes))

    # Runs on CPU.
    (matched_gt_boxes_cpu, matched_gt_classes_cpu,
     matched_gt_indices_cpu, matched_iou_cpu, iou_cpu) = (
         box_ops.box_matching(boxes, gt_boxes, gt_classes))

    # consistency.
    self.assertNDArrayNear(
        matched_gt_boxes_tpu.numpy(), matched_gt_boxes_cpu.numpy(), 1e-5)
    self.assertAllEqual(
        matched_gt_classes_tpu.numpy(), matched_gt_classes_cpu.numpy())
    self.assertAllEqual(
        matched_gt_indices_tpu.numpy(), matched_gt_indices_cpu.numpy())
    self.assertNDArrayNear(
        matched_iou_tpu.numpy(), matched_iou_cpu.numpy(), 1e-5)
    self.assertNDArrayNear(
        iou_tpu.numpy(), iou_cpu.numpy(), 1e-5)

    # correctness.
    self.assertNDArrayNear(
        matched_gt_boxes_tpu.numpy(), matched_gt_boxes_np, 1e-5)
    self.assertAllEqual(
        matched_gt_classes_tpu.numpy(), matched_gt_classes_np)
    self.assertAllEqual(
        matched_gt_indices_tpu.numpy(), matched_gt_indices_np)
    self.assertNDArrayNear(
        matched_iou_tpu.numpy(), matched_iou_np, 1e-5)
    self.assertNDArrayNear(
        iou_tpu.numpy(), iou_np, 1e-5)


if __name__ == '__main__':
  tf.test.main()
