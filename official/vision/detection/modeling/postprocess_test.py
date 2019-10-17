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
"""Tests for postprocess.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from absl.testing import parameterized
from official.vision.detection.dataloader import anchor
from official.vision.detection.modeling import postprocess
from official.modeling.hyperparams import params_dict


class GenerateDetectionsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True),
      (False),
  )
  def testDetectionsOutputShape(self, use_batched_nms):
    min_level = 4
    max_level = 6
    num_scales = 2
    max_total_size = 100
    aspect_ratios = [1.0, 2.0,]
    anchor_scale = 2.0
    output_size = [64, 64]
    num_classes = 4
    # pre_nms_num_boxes = 5000
    score_threshold = 0.01
    batch_size = 1
    postprocessor_params = params_dict.ParamsDict({
        'use_batched_nms': use_batched_nms,
        'max_total_size': max_total_size,
        'nms_iou_threshold': 0.5,
        'score_threshold': score_threshold,
        'min_level': min_level,
        'max_level': max_level,
        'num_classes': num_classes,
    })

    input_anchor = anchor.Anchor(min_level, max_level, num_scales,
                                 aspect_ratios, anchor_scale, output_size)
    cls_outputs_all = (np.random.rand(84, num_classes) -
                       0.5) * 3  # random 84x3 outputs.
    box_outputs_all = np.random.rand(84, 4)  # random 84 boxes.
    class_outputs = {
        4:
            tf.reshape(
                tf.convert_to_tensor(
                    value=cls_outputs_all[0:64], dtype=tf.float32),
                [1, 8, 8, num_classes]),
        5:
            tf.reshape(
                tf.convert_to_tensor(
                    value=cls_outputs_all[64:80], dtype=tf.float32),
                [1, 4, 4, num_classes]),
        6:
            tf.reshape(
                tf.convert_to_tensor(
                    value=cls_outputs_all[80:84], dtype=tf.float32),
                [1, 2, 2, num_classes]),
    }
    box_outputs = {
        4:
            tf.reshape(
                tf.convert_to_tensor(
                    value=box_outputs_all[0:64], dtype=tf.float32),
                [1, 8, 8, 4]),
        5:
            tf.reshape(
                tf.convert_to_tensor(
                    value=box_outputs_all[64:80], dtype=tf.float32),
                [1, 4, 4, 4]),
        6:
            tf.reshape(
                tf.convert_to_tensor(
                    value=box_outputs_all[80:84], dtype=tf.float32),
                [1, 2, 2, 4]),
    }
    image_info = tf.constant([[[1000, 1000], [100, 100], [0.1, 0.1], [0, 0]]],
                             dtype=tf.float32)
    predict_fn = postprocess.GenerateOneStageDetections(postprocessor_params)
    boxes, scores, classes, valid_detections = predict_fn(
        inputs=(box_outputs, class_outputs, input_anchor.multilevel_boxes,
                image_info[:, 1:2, :]))
    (boxes, scores, classes, valid_detections) = [
        boxes.numpy(),
        scores.numpy(),
        classes.numpy(),
        valid_detections.numpy()
    ]
    self.assertEqual(boxes.shape, (batch_size, max_total_size, 4))
    self.assertEqual(scores.shape, (
        batch_size,
        max_total_size,
    ))
    self.assertEqual(classes.shape, (
        batch_size,
        max_total_size,
    ))
    self.assertEqual(valid_detections.shape, (batch_size,))


if __name__ == '__main__':
  assert tf.version.VERSION.startswith('2.')
  tf.test.main()
