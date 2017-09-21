# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.meta_architectures.faster_rcnn_meta_arch."""

import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib


class FasterRCNNMetaArchTest(
    faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase):

  def test_postprocess_second_stage_only_inference_mode_with_masks(self):
    model = self._build_model(
        is_training=False, first_stage_only=False, second_stage_batch_size=6)

    batch_size = 2
    total_num_padded_proposals = batch_size * model.max_num_proposals
    proposal_boxes = tf.constant(
        [[[1, 1, 2, 3],
          [0, 0, 1, 1],
          [.5, .5, .6, .6],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0]],
         [[2, 3, 6, 8],
          [1, 2, 5, 3],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0], 4*[0]]], dtype=tf.float32)
    num_proposals = tf.constant([3, 2], dtype=tf.int32)
    refined_box_encodings = tf.zeros(
        [total_num_padded_proposals, model.num_classes, 4], dtype=tf.float32)
    class_predictions_with_background = tf.ones(
        [total_num_padded_proposals, model.num_classes+1], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 36, 48, 3], dtype=tf.int32)

    mask_height = 2
    mask_width = 2
    mask_predictions = .6 * tf.ones(
        [total_num_padded_proposals, model.num_classes,
         mask_height, mask_width], dtype=tf.float32)
    exp_detection_masks = [[[[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[1, 1], [1, 1]]],
                           [[[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[1, 1], [1, 1]],
                            [[0, 0], [0, 0]]]]

    detections = model.postprocess({
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': proposal_boxes,
        'image_shape': image_shape,
        'mask_predictions': mask_predictions
    })
    with self.test_session() as sess:
      detections_out = sess.run(detections)
      self.assertAllEqual(detections_out['detection_boxes'].shape, [2, 5, 4])
      self.assertAllClose(detections_out['detection_scores'],
                          [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
      self.assertAllClose(detections_out['detection_classes'],
                          [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
      self.assertAllClose(detections_out['num_detections'], [5, 4])
      self.assertAllClose(detections_out['detection_masks'],
                          exp_detection_masks)


if __name__ == '__main__':
  tf.test.main()
