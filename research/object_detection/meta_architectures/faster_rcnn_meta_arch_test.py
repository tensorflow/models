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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib
from object_detection.utils import test_utils


class FasterRCNNMetaArchTest(
    faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase,
    parameterized.TestCase):

  def test_postprocess_second_stage_only_inference_mode_with_masks(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2, second_stage_batch_size=6)

    batch_size = 2
    total_num_padded_proposals = batch_size * model.max_num_proposals
    def graph_fn():
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
      mask_predictions = 30. * tf.ones(
          [total_num_padded_proposals, model.num_classes,
           mask_height, mask_width], dtype=tf.float32)

      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      detections = model.postprocess({
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'num_proposals': num_proposals,
          'proposal_boxes': proposal_boxes,
          'image_shape': image_shape,
          'mask_predictions': mask_predictions
      }, true_image_shapes)
      return (detections['detection_boxes'],
              detections['detection_scores'],
              detections['detection_classes'],
              detections['num_detections'],
              detections['detection_masks'])
    (detection_boxes, detection_scores, detection_classes,
     num_detections, detection_masks) = self.execute_cpu(graph_fn, [], graph=g)
    exp_detection_masks = np.array([[[[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]]],
                                    [[[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[0, 0], [0, 0]]]])
    self.assertAllEqual(detection_boxes.shape, [2, 5, 4])
    self.assertAllClose(detection_scores,
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    self.assertAllClose(detection_classes,
                        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
    self.assertAllClose(num_detections, [5, 4])
    self.assertAllClose(detection_masks, exp_detection_masks)
    self.assertTrue(np.amax(detection_masks <= 1.0))
    self.assertTrue(np.amin(detection_masks >= 0.0))

  def test_postprocess_second_stage_only_inference_mode_with_calibration(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2, second_stage_batch_size=6,
          calibration_mapping_value=0.5)

    batch_size = 2
    total_num_padded_proposals = batch_size * model.max_num_proposals
    def graph_fn():
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
      mask_predictions = 30. * tf.ones(
          [total_num_padded_proposals, model.num_classes,
           mask_height, mask_width], dtype=tf.float32)
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      detections = model.postprocess({
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'num_proposals': num_proposals,
          'proposal_boxes': proposal_boxes,
          'image_shape': image_shape,
          'mask_predictions': mask_predictions
      }, true_image_shapes)
      return (detections['detection_boxes'],
              detections['detection_scores'],
              detections['detection_classes'],
              detections['num_detections'],
              detections['detection_masks'])
    (detection_boxes, detection_scores, detection_classes,
     num_detections, detection_masks) = self.execute_cpu(graph_fn, [], graph=g)
    exp_detection_masks = np.array([[[[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]]],
                                    [[[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]],
                                     [[0, 0], [0, 0]]]])

    self.assertAllEqual(detection_boxes.shape, [2, 5, 4])
    # All scores map to 0.5, except for the final one, which is pruned.
    self.assertAllClose(detection_scores,
                        [[0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5, 0.5, 0.0]])
    self.assertAllClose(detection_classes,
                        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
    self.assertAllClose(num_detections, [5, 4])
    self.assertAllClose(detection_masks,
                        exp_detection_masks)
    self.assertTrue(np.amax(detection_masks <= 1.0))
    self.assertTrue(np.amin(detection_masks >= 0.0))

  def test_postprocess_second_stage_only_inference_mode_with_shared_boxes(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2, second_stage_batch_size=6)

    batch_size = 2
    total_num_padded_proposals = batch_size * model.max_num_proposals
    def graph_fn():
      proposal_boxes = tf.constant(
          [[[1, 1, 2, 3],
            [0, 0, 1, 1],
            [.5, .5, .6, .6],
            4*[0], 4*[0], 4*[0], 4*[0], 4*[0]],
           [[2, 3, 6, 8],
            [1, 2, 5, 3],
            4*[0], 4*[0], 4*[0], 4*[0], 4*[0], 4*[0]]], dtype=tf.float32)
      num_proposals = tf.constant([3, 2], dtype=tf.int32)

      # This has 1 box instead of one for each class.
      refined_box_encodings = tf.zeros(
          [total_num_padded_proposals, 1, 4], dtype=tf.float32)
      class_predictions_with_background = tf.ones(
          [total_num_padded_proposals, model.num_classes+1], dtype=tf.float32)
      image_shape = tf.constant([batch_size, 36, 48, 3], dtype=tf.int32)

      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      detections = model.postprocess({
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
              class_predictions_with_background,
          'num_proposals': num_proposals,
          'proposal_boxes': proposal_boxes,
          'image_shape': image_shape,
      }, true_image_shapes)
      return (detections['detection_boxes'],
              detections['detection_scores'],
              detections['detection_classes'],
              detections['num_detections'])
    (detection_boxes, detection_scores, detection_classes,
     num_detections) = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllEqual(detection_boxes.shape, [2, 5, 4])
    self.assertAllClose(detection_scores,
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    self.assertAllClose(detection_classes,
                        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
    self.assertAllClose(num_detections, [5, 4])

  @parameterized.parameters(
      {'masks_are_class_agnostic': False},
      {'masks_are_class_agnostic': True},
  )
  def test_predict_correct_shapes_in_inference_mode_three_stages_with_masks(
      self, masks_are_class_agnostic):
    batch_size = 2
    image_size = 10
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=3,
          second_stage_batch_size=2,
          predict_masks=True,
          masks_are_class_agnostic=masks_are_class_agnostic)
    def graph_fn():
      shape = [tf.random_uniform([], minval=batch_size, maxval=batch_size + 1,
                                 dtype=tf.int32),
               tf.random_uniform([], minval=image_size, maxval=image_size + 1,
                                 dtype=tf.int32),
               tf.random_uniform([], minval=image_size, maxval=image_size + 1,
                                 dtype=tf.int32),
               3]
      image = tf.zeros(shape)
      _, true_image_shapes = model.preprocess(image)
      detections = model.predict(image, true_image_shapes)
      return (detections['detection_boxes'], detections['detection_classes'],
              detections['detection_scores'], detections['num_detections'],
              detections['detection_masks'], detections['mask_predictions'])
    (detection_boxes, detection_scores, detection_classes,
     num_detections, detection_masks,
     mask_predictions) = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllEqual(detection_boxes.shape, [2, 5, 4])
    self.assertAllEqual(detection_masks.shape,
                        [2, 5, 14, 14])
    self.assertAllEqual(detection_classes.shape, [2, 5])
    self.assertAllEqual(detection_scores.shape, [2, 5])
    self.assertAllEqual(num_detections.shape, [2])
    num_classes = 1 if masks_are_class_agnostic else 2
    self.assertAllEqual(mask_predictions.shape,
                        [10, num_classes, 14, 14])

  def test_raw_detection_boxes_and_anchor_indices_correct(self):
    batch_size = 2
    image_size = 10

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2,
          second_stage_batch_size=2,
          share_box_across_classes=True,
          return_raw_detections_during_predict=True)
    def graph_fn():
      shape = [tf.random_uniform([], minval=batch_size, maxval=batch_size + 1,
                                 dtype=tf.int32),
               tf.random_uniform([], minval=image_size, maxval=image_size + 1,
                                 dtype=tf.int32),
               tf.random_uniform([], minval=image_size, maxval=image_size + 1,
                                 dtype=tf.int32),
               3]
      image = tf.zeros(shape)
      _, true_image_shapes = model.preprocess(image)
      predict_tensor_dict = model.predict(image, true_image_shapes)
      detections = model.postprocess(predict_tensor_dict, true_image_shapes)
      return (detections['detection_boxes'],
              detections['num_detections'],
              detections['detection_anchor_indices'],
              detections['raw_detection_boxes'],
              predict_tensor_dict['raw_detection_boxes'])
    (detection_boxes, num_detections, detection_anchor_indices,
     raw_detection_boxes,
     predict_raw_detection_boxes) = self.execute_cpu(graph_fn, [], graph=g)

    # Verify that the raw detections from predict and postprocess are the
    # same.
    self.assertAllClose(
        np.squeeze(predict_raw_detection_boxes), raw_detection_boxes)
    # Verify that the raw detection boxes at detection anchor indices are the
    # same as the postprocessed detections.
    for i in range(batch_size):
      num_detections_per_image = int(num_detections[i])
      detection_boxes_per_image = detection_boxes[i][
          :num_detections_per_image]
      detection_anchor_indices_per_image = detection_anchor_indices[i][
          :num_detections_per_image]
      raw_detections_per_image = np.squeeze(raw_detection_boxes[i])
      raw_detections_at_anchor_indices = raw_detections_per_image[
          detection_anchor_indices_per_image]
      self.assertAllClose(detection_boxes_per_image,
                          raw_detections_at_anchor_indices)

  @parameterized.parameters(
      {'masks_are_class_agnostic': False},
      {'masks_are_class_agnostic': True},
  )
  def test_predict_gives_correct_shapes_in_train_mode_both_stages_with_masks(
      self, masks_are_class_agnostic):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=True,
          number_of_stages=3,
          second_stage_batch_size=7,
          predict_masks=True,
          masks_are_class_agnostic=masks_are_class_agnostic)
    batch_size = 2
    image_size = 10
    max_num_proposals = 7
    def graph_fn():
      image_shape = (batch_size, image_size, image_size, 3)
      preprocessed_inputs = tf.zeros(image_shape, dtype=tf.float32)
      groundtruth_boxes_list = [
          tf.constant([[0, 0, .5, .5], [.5, .5, 1, 1]], dtype=tf.float32),
          tf.constant([[0, .5, .5, 1], [.5, 0, 1, .5]], dtype=tf.float32)
      ]
      groundtruth_classes_list = [
          tf.constant([[1, 0], [0, 1]], dtype=tf.float32),
          tf.constant([[1, 0], [1, 0]], dtype=tf.float32)
      ]
      groundtruth_weights_list = [
          tf.constant([1, 1], dtype=tf.float32),
          tf.constant([1, 1], dtype=tf.float32)]
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      model.provide_groundtruth(
          groundtruth_boxes_list,
          groundtruth_classes_list,
          groundtruth_weights_list=groundtruth_weights_list)

      result_tensor_dict = model.predict(preprocessed_inputs, true_image_shapes)
      return result_tensor_dict['mask_predictions']
    mask_shape_1 = 1 if masks_are_class_agnostic else model._num_classes
    mask_out = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllEqual(mask_out.shape,
                        (2 * max_num_proposals, mask_shape_1, 14, 14))

  def test_postprocess_third_stage_only_inference_mode(self):
    batch_size = 2
    initial_crop_size = 3
    maxpool_stride = 1
    height = initial_crop_size // maxpool_stride
    width = initial_crop_size // maxpool_stride
    depth = 3

    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False, number_of_stages=3,
          second_stage_batch_size=6, predict_masks=True)
    total_num_padded_proposals = batch_size * model.max_num_proposals
    def graph_fn(images_shape, num_proposals, proposal_boxes,
                 refined_box_encodings, class_predictions_with_background):
      _, true_image_shapes = model.preprocess(
          tf.zeros(images_shape))
      detections = model.postprocess({
          'refined_box_encodings': refined_box_encodings,
          'class_predictions_with_background':
          class_predictions_with_background,
          'num_proposals': num_proposals,
          'proposal_boxes': proposal_boxes,
          'image_shape': images_shape,
          'detection_boxes': tf.zeros([2, 5, 4]),
          'detection_masks': tf.zeros([2, 5, 14, 14]),
          'detection_scores': tf.zeros([2, 5]),
          'detection_classes': tf.zeros([2, 5]),
          'num_detections': tf.zeros([2]),
          'detection_features': tf.zeros([2, 5, width, height, depth])
      }, true_image_shapes)
      return (detections['detection_boxes'], detections['detection_masks'],
              detections['detection_scores'], detections['detection_classes'],
              detections['num_detections'],
              detections['detection_features'])
    images_shape = np.array((2, 36, 48, 3), dtype=np.int32)
    proposal_boxes = np.array(
        [[[1, 1, 2, 3],
          [0, 0, 1, 1],
          [.5, .5, .6, .6],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0]],
         [[2, 3, 6, 8],
          [1, 2, 5, 3],
          4*[0], 4*[0], 4*[0], 4*[0], 4*[0], 4*[0]]])
    num_proposals = np.array([3, 2], dtype=np.int32)
    refined_box_encodings = np.zeros(
        [total_num_padded_proposals, model.num_classes, 4])
    class_predictions_with_background = np.ones(
        [total_num_padded_proposals, model.num_classes+1])

    (detection_boxes, detection_masks, detection_scores, detection_classes,
     num_detections,
     detection_features) = self.execute_cpu(graph_fn,
                                            [images_shape, num_proposals,
                                             proposal_boxes,
                                             refined_box_encodings,
                                             class_predictions_with_background],
                                            graph=g)
    self.assertAllEqual(detection_boxes.shape, [2, 5, 4])
    self.assertAllEqual(detection_masks.shape, [2, 5, 14, 14])
    self.assertAllClose(detection_scores.shape, [2, 5])
    self.assertAllClose(detection_classes.shape, [2, 5])
    self.assertAllClose(num_detections.shape, [2])
    self.assertTrue(np.amax(detection_masks <= 1.0))
    self.assertTrue(np.amin(detection_masks >= 0.0))
    self.assertAllEqual(detection_features.shape,
                        [2, 5, width, height, depth])
    self.assertGreaterEqual(np.amax(detection_features), 0)

  def _get_box_classifier_features_shape(self,
                                         image_size,
                                         batch_size,
                                         max_num_proposals,
                                         initial_crop_size,
                                         maxpool_stride,
                                         num_features):
    return (batch_size * max_num_proposals,
            initial_crop_size // maxpool_stride,
            initial_crop_size // maxpool_stride,
            num_features)

  def test_output_final_box_features(self):
    with test_utils.GraphContextOrNone() as g:
      model = self._build_model(
          is_training=False,
          number_of_stages=2,
          second_stage_batch_size=6,
          output_final_box_features=True)

    batch_size = 2
    total_num_padded_proposals = batch_size * model.max_num_proposals
    def graph_fn():
      proposal_boxes = tf.constant([[[1, 1, 2, 3], [0, 0, 1, 1],
                                     [.5, .5, .6, .6], 4 * [0], 4 * [0],
                                     4 * [0], 4 * [0], 4 * [0]],
                                    [[2, 3, 6, 8], [1, 2, 5, 3], 4 * [0],
                                     4 * [0], 4 * [0], 4 * [0], 4 * [0],
                                     4 * [0]]],
                                   dtype=tf.float32)
      num_proposals = tf.constant([3, 2], dtype=tf.int32)
      refined_box_encodings = tf.zeros(
          [total_num_padded_proposals, model.num_classes, 4], dtype=tf.float32)
      class_predictions_with_background = tf.ones(
          [total_num_padded_proposals, model.num_classes + 1], dtype=tf.float32)
      image_shape = tf.constant([batch_size, 36, 48, 3], dtype=tf.int32)

      mask_height = 2
      mask_width = 2
      mask_predictions = 30. * tf.ones([
          total_num_padded_proposals, model.num_classes, mask_height, mask_width
      ],
                                       dtype=tf.float32)
      _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
      rpn_features_to_crop = tf.ones((batch_size, mask_height, mask_width, 3),
                                     tf.float32)
      detections = model.postprocess(
          {
              'refined_box_encodings':
                  refined_box_encodings,
              'class_predictions_with_background':
                  class_predictions_with_background,
              'num_proposals':
                  num_proposals,
              'proposal_boxes':
                  proposal_boxes,
              'image_shape':
                  image_shape,
              'mask_predictions':
                  mask_predictions,
              'rpn_features_to_crop':
                  [rpn_features_to_crop]
          }, true_image_shapes)
      self.assertIn('detection_features', detections)
      return (detections['detection_boxes'], detections['detection_scores'],
              detections['detection_classes'], detections['num_detections'],
              detections['detection_masks'])
    (detection_boxes, detection_scores, detection_classes, num_detections,
     detection_masks) = self.execute_cpu(graph_fn, [], graph=g)
    exp_detection_masks = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]]],
                                    [[[1, 1], [1, 1]], [[1, 1], [1, 1]],
                                     [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                                     [[0, 0], [0, 0]]]])

    self.assertAllEqual(detection_boxes.shape, [2, 5, 4])
    self.assertAllClose(detection_scores,
                        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    self.assertAllClose(detection_classes,
                        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
    self.assertAllClose(num_detections, [5, 4])
    self.assertAllClose(detection_masks,
                        exp_detection_masks)


if __name__ == '__main__':
  tf.test.main()
