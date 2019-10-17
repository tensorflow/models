# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for box_utils.py."""

import numpy as np
import tensorflow.compat.v2 as tf
from official.vision.detection.utils import box_utils


def _transform_boxes_on_tpu_and_cpu(transform_fn, boxes, arg):
  # Runs on TPU.
  strategy = tf.distribute.experimental.TPUStrategy()
  with strategy.scope():
    normalized_op_tpu = transform_fn(boxes, arg)
    normalized_boxes_tpu = normalized_op_tpu.numpy()

  # Runs on CPU.
  normalize_op = transform_fn(boxes, arg)
  normalized_boxes_cpu = normalize_op.numpy()
  return normalized_boxes_tpu, normalized_boxes_cpu


class NormalizeBoxesTest(tf.test.TestCase):

  def testNormalizeBoxes1DWithImageShapeAsList(self):
    boxes = tf.constant([10, 30, 40, 90], tf.float32)
    image_shape = [50, 100]

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [0.2, 0.3, 0.8, 0.9], 1e-5)

  def testNormalizeBoxes1DWithImageShapeAsTensor(self):
    boxes = tf.constant([10, 30, 40, 90], tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [0.2, 0.3, 0.8, 0.9], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsList(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = [50, 100]

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsVector(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = tf.constant([50, 100], dtype=tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = tf.constant([[50, 100]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes2DWithImageShapeAsSameShapeTensor(self):
    boxes = tf.constant([[10, 30, 40, 90], [30, 10, 40, 50]], tf.float32)
    image_shape = tf.constant([[50, 100], [50, 100]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsList(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = [50, 100]

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                            [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsVector(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                            [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]], 1e-5)

  def testNormalizeBoxes3DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[[10, 30, 40, 90], [30, 10, 40, 50]],
                         [[20, 40, 50, 80], [30, 50, 40, 90]]], tf.float32)
    image_shape = tf.constant([[[50, 100]], [[500, 1000]]], tf.int32)

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

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

    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.normalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(
        normalized_boxes_tpu,
        [[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
         [[0.04, 0.04, 0.1, 0.08], [0.06, 0.05, 0.08, 0.09]]], 1e-5)


class DenormalizeBoxesTest(tf.test.TestCase):

  def testDenormalizeBoxes1DWithImageShapeAsList(self):
    boxes = tf.constant([0.2, 0.3, 0.8, 0.9], tf.float32)
    image_shape = [50, 100]
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [10, 30, 40, 90], 1e-5)

  def testDenormalizeBoxes1DWithImageShapeAsTensor(self):
    boxes = tf.constant([0.2, 0.3, 0.8, 0.9], tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu, [10, 30, 40, 90], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsList(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = [50, 100]
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsVector(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = tf.constant([50, 100], dtype=tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = tf.constant([[50, 100]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes2DWithImageShapeAsSameShapeTensor(self):
    boxes = tf.constant([[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                        tf.float32)
    image_shape = tf.constant([[50, 100], [50, 100]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[10, 30, 40, 90], [30, 10, 40, 50]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsList(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]],
                        tf.float32)
    image_shape = [50, 100]
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[10, 30, 40, 90], [30, 10, 40, 50]],
                            [[20, 40, 50, 80], [30, 50, 40, 90]]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsVector(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.4, 0.4, 1.0, 0.8], [0.6, 0.5, 0.8, 0.9]]],
                        tf.float32)
    image_shape = tf.constant([50, 100], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

    self.assertNDArrayNear(normalized_boxes_tpu, normalized_boxes_cpu, 1e-5)
    self.assertNDArrayNear(normalized_boxes_tpu,
                           [[[10, 30, 40, 90], [30, 10, 40, 50]],
                            [[20, 40, 50, 80], [30, 50, 40, 90]]], 1e-5)

  def testDenormalizeBoxes3DWithImageShapeAsBroadcastableTensor(self):
    boxes = tf.constant([[[0.2, 0.3, 0.8, 0.9], [0.6, 0.1, 0.8, 0.5]],
                         [[0.04, 0.04, 0.1, 0.08], [0.06, 0.05, 0.08, 0.09]]],
                        tf.float32)
    image_shape = tf.constant([[[50, 100]], [[500, 1000]]], tf.int32)
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

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
    normalized_boxes_tpu, normalized_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        box_utils.denormalize_boxes, boxes, image_shape)

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
        box_utils.clip_boxes, boxes, image_shape)

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
        box_utils.clip_boxes, boxes, image_shape)

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
        box_utils.clip_boxes, boxes, image_shape)

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
      encoded_boxes = box_utils.encode_boxes(boxes, anchors, weights)
      decoded_boxes = box_utils.decode_boxes(encoded_boxes, anchors, weights)
      return decoded_boxes

    decoded_boxes_tpu, decoded_boxes_cpu = _transform_boxes_on_tpu_and_cpu(
        test_fn, boxes, anchors)

    self.assertNDArrayNear(decoded_boxes_tpu, decoded_boxes_cpu, 1e-5)
    self.assertNDArrayNear(decoded_boxes_tpu, boxes_np, 1e-5)


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
