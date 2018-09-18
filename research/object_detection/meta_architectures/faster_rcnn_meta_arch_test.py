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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch_test_lib


class FasterRCNNMetaArchTest(
    faster_rcnn_meta_arch_test_lib.FasterRCNNMetaArchTestBase,
    parameterized.TestCase):

  def test_postprocess_second_stage_only_inference_mode_with_masks(self):
    model = self._build_model(
        is_training=False, number_of_stages=2, second_stage_batch_size=6)

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
    mask_predictions = 30. * tf.ones(
        [total_num_padded_proposals, model.num_classes,
         mask_height, mask_width], dtype=tf.float32)
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

    _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
    detections = model.postprocess({
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': proposal_boxes,
        'image_shape': image_shape,
        'mask_predictions': mask_predictions
    }, true_image_shapes)
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
      self.assertTrue(np.amax(detections_out['detection_masks'] <= 1.0))
      self.assertTrue(np.amin(detections_out['detection_masks'] >= 0.0))

  def test_postprocess_second_stage_only_inference_mode_with_shared_boxes(self):
    model = self._build_model(
        is_training=False, number_of_stages=2, second_stage_batch_size=6)

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

    # This has 1 box instead of one for each class.
    refined_box_encodings = tf.zeros(
        [total_num_padded_proposals, 1, 4], dtype=tf.float32)
    class_predictions_with_background = tf.ones(
        [total_num_padded_proposals, model.num_classes+1], dtype=tf.float32)
    image_shape = tf.constant([batch_size, 36, 48, 3], dtype=tf.int32)

    _, true_image_shapes = model.preprocess(tf.zeros(image_shape))
    detections = model.postprocess({
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background': class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': proposal_boxes,
        'image_shape': image_shape,
    }, true_image_shapes)
    with self.test_session() as sess:
      detections_out = sess.run(detections)
      self.assertAllEqual(detections_out['detection_boxes'].shape, [2, 5, 4])
      self.assertAllClose(detections_out['detection_scores'],
                          [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
      self.assertAllClose(detections_out['detection_classes'],
                          [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]])
      self.assertAllClose(detections_out['num_detections'], [5, 4])

  @parameterized.parameters(
      {'masks_are_class_agnostic': False},
      {'masks_are_class_agnostic': True},
  )
  def test_predict_correct_shapes_in_inference_mode_three_stages_with_masks(
      self, masks_are_class_agnostic):
    batch_size = 2
    image_size = 10
    max_num_proposals = 8
    initial_crop_size = 3
    maxpool_stride = 1

    input_shapes = [(batch_size, image_size, image_size, 3),
                    (None, image_size, image_size, 3),
                    (batch_size, None, None, 3),
                    (None, None, None, 3)]
    expected_num_anchors = image_size * image_size * 3 * 3
    expected_shapes = {
        'rpn_box_predictor_features':
        (2, image_size, image_size, 512),
        'rpn_features_to_crop': (2, image_size, image_size, 3),
        'image_shape': (4,),
        'rpn_box_encodings': (2, expected_num_anchors, 4),
        'rpn_objectness_predictions_with_background':
        (2, expected_num_anchors, 2),
        'anchors': (expected_num_anchors, 4),
        'refined_box_encodings': (2 * max_num_proposals, 2, 4),
        'class_predictions_with_background': (2 * max_num_proposals, 2 + 1),
        'num_proposals': (2,),
        'proposal_boxes': (2, max_num_proposals, 4),
        'proposal_boxes_normalized': (2, max_num_proposals, 4),
        'box_classifier_features':
        self._get_box_classifier_features_shape(image_size,
                                                batch_size,
                                                max_num_proposals,
                                                initial_crop_size,
                                                maxpool_stride,
                                                3)
    }

    for input_shape in input_shapes:
      test_graph = tf.Graph()
      with test_graph.as_default():
        model = self._build_model(
            is_training=False,
            number_of_stages=3,
            second_stage_batch_size=2,
            predict_masks=True,
            masks_are_class_agnostic=masks_are_class_agnostic)
        preprocessed_inputs = tf.placeholder(tf.float32, shape=input_shape)
        _, true_image_shapes = model.preprocess(preprocessed_inputs)
        result_tensor_dict = model.predict(preprocessed_inputs,
                                           true_image_shapes)
        init_op = tf.global_variables_initializer()
      with self.test_session(graph=test_graph) as sess:
        sess.run(init_op)
        tensor_dict_out = sess.run(result_tensor_dict, feed_dict={
            preprocessed_inputs:
            np.zeros((batch_size, image_size, image_size, 3))})
      self.assertEqual(
          set(tensor_dict_out.keys()),
          set(expected_shapes.keys()).union(
              set([
                  'detection_boxes', 'detection_scores', 'detection_classes',
                  'detection_masks', 'num_detections'
              ])))
      for key in expected_shapes:
        self.assertAllEqual(tensor_dict_out[key].shape, expected_shapes[key])
      self.assertAllEqual(tensor_dict_out['detection_boxes'].shape, [2, 5, 4])
      self.assertAllEqual(tensor_dict_out['detection_masks'].shape,
                          [2, 5, 14, 14])
      self.assertAllEqual(tensor_dict_out['detection_classes'].shape, [2, 5])
      self.assertAllEqual(tensor_dict_out['detection_scores'].shape, [2, 5])
      self.assertAllEqual(tensor_dict_out['num_detections'].shape, [2])

  @parameterized.parameters(
      {'masks_are_class_agnostic': False},
      {'masks_are_class_agnostic': True},
  )
  def test_predict_gives_correct_shapes_in_train_mode_both_stages_with_masks(
      self, masks_are_class_agnostic):
    test_graph = tf.Graph()
    with test_graph.as_default():
      model = self._build_model(
          is_training=True,
          number_of_stages=3,
          second_stage_batch_size=7,
          predict_masks=True,
          masks_are_class_agnostic=masks_are_class_agnostic)
      batch_size = 2
      image_size = 10
      max_num_proposals = 7
      initial_crop_size = 3
      maxpool_stride = 1

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
      mask_shape_1 = 1 if masks_are_class_agnostic else model._num_classes
      expected_shapes = {
          'rpn_box_predictor_features': (2, image_size, image_size, 512),
          'rpn_features_to_crop': (2, image_size, image_size, 3),
          'image_shape': (4,),
          'refined_box_encodings': (2 * max_num_proposals, 2, 4),
          'class_predictions_with_background': (2 * max_num_proposals, 2 + 1),
          'num_proposals': (2,),
          'proposal_boxes': (2, max_num_proposals, 4),
          'proposal_boxes_normalized': (2, max_num_proposals, 4),
          'box_classifier_features':
              self._get_box_classifier_features_shape(
                  image_size, batch_size, max_num_proposals, initial_crop_size,
                  maxpool_stride, 3),
          'mask_predictions': (2 * max_num_proposals, mask_shape_1, 14, 14)
      }

      init_op = tf.global_variables_initializer()
      with self.test_session(graph=test_graph) as sess:
        sess.run(init_op)
        tensor_dict_out = sess.run(result_tensor_dict)
        self.assertEqual(
            set(tensor_dict_out.keys()),
            set(expected_shapes.keys()).union(
                set([
                    'rpn_box_encodings',
                    'rpn_objectness_predictions_with_background',
                    'anchors',
                ])))
        for key in expected_shapes:
          self.assertAllEqual(tensor_dict_out[key].shape, expected_shapes[key])

        anchors_shape_out = tensor_dict_out['anchors'].shape
        self.assertEqual(2, len(anchors_shape_out))
        self.assertEqual(4, anchors_shape_out[1])
        num_anchors_out = anchors_shape_out[0]
        self.assertAllEqual(tensor_dict_out['rpn_box_encodings'].shape,
                            (2, num_anchors_out, 4))
        self.assertAllEqual(
            tensor_dict_out['rpn_objectness_predictions_with_background'].shape,
            (2, num_anchors_out, 2))

  def test_postprocess_third_stage_only_inference_mode(self):
    num_proposals_shapes = [(2), (None)]
    refined_box_encodings_shapes = [(16, 2, 4), (None, 2, 4)]
    class_predictions_with_background_shapes = [(16, 3), (None, 3)]
    proposal_boxes_shapes = [(2, 8, 4), (None, 8, 4)]
    batch_size = 2
    image_shape = np.array((2, 36, 48, 3), dtype=np.int32)
    for (num_proposals_shape, refined_box_encoding_shape,
         class_predictions_with_background_shape,
         proposal_boxes_shape) in zip(num_proposals_shapes,
                                      refined_box_encodings_shapes,
                                      class_predictions_with_background_shapes,
                                      proposal_boxes_shapes):
      tf_graph = tf.Graph()
      with tf_graph.as_default():
        model = self._build_model(
            is_training=False, number_of_stages=3,
            second_stage_batch_size=6, predict_masks=True)
        total_num_padded_proposals = batch_size * model.max_num_proposals
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

        num_proposals_placeholder = tf.placeholder(tf.int32,
                                                   shape=num_proposals_shape)
        refined_box_encodings_placeholder = tf.placeholder(
            tf.float32, shape=refined_box_encoding_shape)
        class_predictions_with_background_placeholder = tf.placeholder(
            tf.float32, shape=class_predictions_with_background_shape)
        proposal_boxes_placeholder = tf.placeholder(
            tf.float32, shape=proposal_boxes_shape)
        image_shape_placeholder = tf.placeholder(tf.int32, shape=(4))
        _, true_image_shapes = model.preprocess(
            tf.zeros(image_shape_placeholder))
        detections = model.postprocess({
            'refined_box_encodings': refined_box_encodings_placeholder,
            'class_predictions_with_background':
            class_predictions_with_background_placeholder,
            'num_proposals': num_proposals_placeholder,
            'proposal_boxes': proposal_boxes_placeholder,
            'image_shape': image_shape_placeholder,
            'detection_boxes': tf.zeros([2, 5, 4]),
            'detection_masks': tf.zeros([2, 5, 14, 14]),
            'detection_scores': tf.zeros([2, 5]),
            'detection_classes': tf.zeros([2, 5]),
            'num_detections': tf.zeros([2]),
        }, true_image_shapes)
      with self.test_session(graph=tf_graph) as sess:
        detections_out = sess.run(
            detections,
            feed_dict={
                refined_box_encodings_placeholder: refined_box_encodings,
                class_predictions_with_background_placeholder:
                class_predictions_with_background,
                num_proposals_placeholder: num_proposals,
                proposal_boxes_placeholder: proposal_boxes,
                image_shape_placeholder: image_shape
            })
      self.assertAllEqual(detections_out['detection_boxes'].shape, [2, 5, 4])
      self.assertAllEqual(detections_out['detection_masks'].shape,
                          [2, 5, 14, 14])
      self.assertAllClose(detections_out['detection_scores'].shape, [2, 5])
      self.assertAllClose(detections_out['detection_classes'].shape, [2, 5])
      self.assertAllClose(detections_out['num_detections'].shape, [2])
      self.assertTrue(np.amax(detections_out['detection_masks'] <= 1.0))
      self.assertTrue(np.amin(detections_out['detection_masks'] >= 0.0))

  def _get_box_classifier_features_shape(self,
                                         image_size,
                                         batch_size,
                                         max_num_proposals,
                                         initial_crop_size,
                                         maxpool_stride,
                                         num_features):
    return (batch_size * max_num_proposals,
            initial_crop_size/maxpool_stride,
            initial_crop_size/maxpool_stride,
            num_features)

if __name__ == '__main__':
  tf.test.main()
