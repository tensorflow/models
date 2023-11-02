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

"""Tests for yolo heads."""

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.projects.yolo.losses import yolov7_loss
from official.projects.yolo.ops import box_ops

_HEIGHT, _WIDTH = 640, 640
_BATCH_SIZE = 8
_NUM_GTS = 100
_NUM_LAYERS, _NUM_ANCHORS = 3, 3
_NUM_CLASSES = 80


def build_labels():
  image_info = tf.constant(
      [
          [[640, 640], [640, 640], [1.0, 1.0], [0.0, 0.0]]
          for _ in range(_BATCH_SIZE)
      ], dtype=tf.float32
  )
  box_y1x1 = np.random.rand(_BATCH_SIZE, _NUM_GTS, 2).astype(np.float32)
  box_y2x2 = (
      np.random.rand(_BATCH_SIZE, _NUM_GTS, 2).astype(np.float32)
      * (1 - box_y1x1)
      + box_y1x1
  )
  boxes_yxyx = tf.concat([box_y1x1, box_y2x2], axis=-1)
  num_detections = np.random.randint(_NUM_GTS, size=[_BATCH_SIZE])
  classes = np.arange(_NUM_GTS * _BATCH_SIZE).reshape([_BATCH_SIZE, -1])
  for i in range(_BATCH_SIZE):
    classes[i, num_detections[i]:] = -1
  classes = tf.constant(classes, dtype=tf.int32)
  return {'image_info': image_info, 'classes': classes, 'bbox': boxes_yxyx}


def build_predictions():
  # Scale down by 2^3 because prediction outputs start at level 3.
  h, w = _HEIGHT // 8, _WIDTH // 8

  predictions = {}
  for i in range(_NUM_LAYERS):
    shape = [_BATCH_SIZE, h // (2**i), w // (2**i), _NUM_ANCHORS]
    p_y1x1 = tf.constant(np.random.rand(*shape, 2), dtype=tf.float32)
    p_y2x2 = tf.constant(np.random.rand(*shape, 2), dtype=tf.float32)
    # Transform the box from yxyx to xywh.
    p_box = box_ops.yxyx_to_xcycwh(tf.concat([p_y1x1, p_y2x2], axis=-1))
    p_obj = tf.constant(np.random.rand(*shape, 1), dtype=tf.float32)
    p_cls = tf.constant(np.random.rand(*shape, _NUM_CLASSES), dtype=tf.float32)
    predictions[str(i + 3)] = tf.concat([p_box, p_obj, p_cls], axis=-1)

  return predictions


class YoloV7LossTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(42)
    self._anchors = [
        [[12, 16], [19, 36], [40, 28]],  # Level 3
        [[36, 75], [76, 55], [72, 146]],  # Level 4
        [[142, 110], [192, 243], [459, 401]],  # Level 5
    ]
    self._strides = [8, 16, 32]

  @parameterized.product(
      gamma=(0.0, 1.5), label_smoothing=(0.0, 0.2), auto_balance=(True, False)
  )
  def test_loss(self, gamma, label_smoothing, auto_balance):
    """Test YOLOv7 normal loss."""
    labels = build_labels()
    predictions = build_predictions()
    loss = yolov7_loss.YoloV7Loss(
        anchors=self._anchors,
        strides=self._strides,
        input_size=[_HEIGHT, _WIDTH],
        gamma=gamma,
        label_smoothing=label_smoothing,
        num_classes=_NUM_CLASSES,
        auto_balance=auto_balance,
    )

    loss_val = loss(labels, predictions)
    losses = loss.report_separate_losses()
    logging.info('loss_val: %.6f', loss_val)
    logging.info('box_loss: %.6f', losses['box_loss'])
    logging.info('obj_loss: %.6f', losses['obj_loss'])
    logging.info('cls_loss: %.6f', losses['cls_loss'])

    expected_loss_val = (
        losses['box_loss'] + losses['obj_loss'] + losses['cls_loss']
    ) * _BATCH_SIZE
    self.assertNear(loss_val, expected_loss_val, err=1e-6)

  @parameterized.product(
      gamma=(0.0, 1.5), label_smoothing=(0.0, 0.2), auto_balance=(True, False)
  )
  def test_loss_ota(self, gamma, label_smoothing, auto_balance):
    """Test YOLOv7 OTA loss."""
    labels = build_labels()
    predictions = build_predictions()
    loss = yolov7_loss.YoloV7LossOTA(
        anchors=self._anchors,
        strides=self._strides,
        input_size=[_HEIGHT, _WIDTH],
        gamma=gamma,
        label_smoothing=label_smoothing,
        num_classes=_NUM_CLASSES,
        auto_balance=auto_balance,
    )

    loss_val = loss(labels, predictions)
    losses = loss.report_separate_losses()
    logging.info('loss_val: %.6f', loss_val)
    logging.info('box_loss: %.6f', losses['box_loss'])
    logging.info('obj_loss: %.6f', losses['obj_loss'])
    logging.info('cls_loss: %.6f', losses['cls_loss'])

    expected_loss_val = (
        losses['box_loss'] + losses['obj_loss'] + losses['cls_loss']
    ) * _BATCH_SIZE
    self.assertNear(loss_val, expected_loss_val, err=1e-6)


if __name__ == '__main__':
  tf.test.main()
