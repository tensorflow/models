# Lint as: python2, python3
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

"""Tests for object_detection.meta_architectures.ssd_meta_arch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.meta_architectures import ssd_meta_arch_test_lib
from object_detection.protos import model_pb2
from object_detection.utils import test_utils

# pylint: disable=g-import-not-at-top
try:
  import tf_slim as slim
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top

keras = tf.keras.layers


class SsdMetaArchTest(ssd_meta_arch_test_lib.SSDMetaArchTestBase,
                      parameterized.TestCase):

  def _create_model(
      self,
      apply_hard_mining=True,
      normalize_loc_loss_by_codesize=False,
      add_background_class=True,
      random_example_sampling=False,
      expected_loss_weights=model_pb2.DetectionModel().ssd.loss.NONE,
      min_num_negative_samples=1,
      desired_negative_sampling_ratio=3,
      predict_mask=False,
      use_static_shapes=False,
      nms_max_size_per_class=5,
      calibration_mapping_value=None,
      return_raw_detections_during_predict=False):
    return super(SsdMetaArchTest, self)._create_model(
        model_fn=ssd_meta_arch.SSDMetaArch,
        apply_hard_mining=apply_hard_mining,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        add_background_class=add_background_class,
        random_example_sampling=random_example_sampling,
        expected_loss_weights=expected_loss_weights,
        min_num_negative_samples=min_num_negative_samples,
        desired_negative_sampling_ratio=desired_negative_sampling_ratio,
        predict_mask=predict_mask,
        use_static_shapes=use_static_shapes,
        nms_max_size_per_class=nms_max_size_per_class,
        calibration_mapping_value=calibration_mapping_value,
        return_raw_detections_during_predict=(
            return_raw_detections_during_predict))

  def test_preprocess_preserves_shapes_with_dynamic_input_image(self):
    width = tf.random.uniform([], minval=5, maxval=10, dtype=tf.int32)
    batch = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
    shape = tf.stack([batch, 5, width, 3])
    image = tf.random.uniform(shape)
    model, _, _, _ = self._create_model()
    preprocessed_inputs, _ = model.preprocess(image)
    self.assertTrue(
        preprocessed_inputs.shape.is_compatible_with([None, 5, None, 3]))

  def test_preprocess_preserves_shape_with_static_input_image(self):
    image = tf.random.uniform([2, 3, 3, 3])
    model, _, _, _ = self._create_model()
    preprocessed_inputs, _ = model.preprocess(image)
    self.assertTrue(preprocessed_inputs.shape.is_compatible_with([2, 3, 3, 3]))

  def test_predict_result_shapes_on_image_with_dynamic_shape(self):
    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, code_size = self._create_model()

    def graph_fn():
      size = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      batch = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      shape = tf.stack([batch, size, size, 3])
      image = tf.random.uniform(shape)
      prediction_dict = model.predict(image, true_image_shapes=None)
      self.assertIn('box_encodings', prediction_dict)
      self.assertIn('class_predictions_with_background', prediction_dict)
      self.assertIn('feature_maps', prediction_dict)
      self.assertIn('anchors', prediction_dict)
      self.assertIn('final_anchors', prediction_dict)
      return (prediction_dict['box_encodings'],
              prediction_dict['final_anchors'],
              prediction_dict['class_predictions_with_background'],
              tf.constant(num_anchors), batch)
    (box_encodings_out, final_anchors, class_predictions_with_background,
     num_anchors, batch_size) = self.execute_cpu(graph_fn, [], graph=g)
    self.assertAllEqual(box_encodings_out.shape,
                        (batch_size, num_anchors, code_size))
    self.assertAllEqual(final_anchors.shape,
                        (batch_size, num_anchors, code_size))
    self.assertAllEqual(
        class_predictions_with_background.shape,
        (batch_size, num_anchors, num_classes + 1))

  def test_predict_result_shapes_on_image_with_static_shape(self):

    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, code_size = self._create_model()

    def graph_fn(input_image):
      predictions = model.predict(input_image, true_image_shapes=None)
      return (predictions['box_encodings'],
              predictions['class_predictions_with_background'],
              predictions['final_anchors'])
    batch_size = 3
    image_size = 2
    channels = 3
    input_image = np.random.rand(batch_size, image_size, image_size,
                                 channels).astype(np.float32)
    expected_box_encodings_shape = (batch_size, num_anchors, code_size)
    expected_class_predictions_shape = (batch_size, num_anchors, num_classes+1)
    final_anchors_shape = (batch_size, num_anchors, 4)
    (box_encodings, class_predictions, final_anchors) = self.execute(
        graph_fn, [input_image], graph=g)
    self.assertAllEqual(box_encodings.shape, expected_box_encodings_shape)
    self.assertAllEqual(class_predictions.shape,
                        expected_class_predictions_shape)
    self.assertAllEqual(final_anchors.shape, final_anchors_shape)

  def test_predict_with_raw_output_fields(self):
    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, code_size = self._create_model(
          return_raw_detections_during_predict=True)

    def graph_fn(input_image):
      predictions = model.predict(input_image, true_image_shapes=None)
      return (predictions['box_encodings'],
              predictions['class_predictions_with_background'],
              predictions['final_anchors'],
              predictions['raw_detection_boxes'],
              predictions['raw_detection_feature_map_indices'])
    batch_size = 3
    image_size = 2
    channels = 3
    input_image = np.random.rand(batch_size, image_size, image_size,
                                 channels).astype(np.float32)
    expected_box_encodings_shape = (batch_size, num_anchors, code_size)
    expected_class_predictions_shape = (batch_size, num_anchors, num_classes+1)
    final_anchors_shape = (batch_size, num_anchors, 4)
    expected_raw_detection_boxes_shape = (batch_size, num_anchors, 4)
    (box_encodings, class_predictions, final_anchors, raw_detection_boxes,
     raw_detection_feature_map_indices) = self.execute(
         graph_fn, [input_image], graph=g)
    self.assertAllEqual(box_encodings.shape, expected_box_encodings_shape)
    self.assertAllEqual(class_predictions.shape,
                        expected_class_predictions_shape)
    self.assertAllEqual(final_anchors.shape, final_anchors_shape)
    self.assertAllEqual(raw_detection_boxes.shape,
                        expected_raw_detection_boxes_shape)
    self.assertAllEqual(raw_detection_feature_map_indices,
                        np.zeros((batch_size, num_anchors)))

  def test_raw_detection_boxes_agree_predict_postprocess(self):
    with test_utils.GraphContextOrNone() as g:
      model, _, _, _ = self._create_model(
          return_raw_detections_during_predict=True)
    def graph_fn():
      size = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      batch = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      shape = tf.stack([batch, size, size, 3])
      image = tf.random.uniform(shape)
      preprocessed_inputs, true_image_shapes = model.preprocess(
          image)
      prediction_dict = model.predict(preprocessed_inputs,
                                      true_image_shapes)
      raw_detection_boxes_predict = prediction_dict['raw_detection_boxes']
      detections = model.postprocess(prediction_dict, true_image_shapes)
      raw_detection_boxes_postprocess = detections['raw_detection_boxes']
      return raw_detection_boxes_predict, raw_detection_boxes_postprocess
    (raw_detection_boxes_predict_out,
     raw_detection_boxes_postprocess_out) = self.execute_cpu(graph_fn, [],
                                                             graph=g)
    self.assertAllEqual(raw_detection_boxes_predict_out,
                        raw_detection_boxes_postprocess_out)

  def test_postprocess_results_are_correct(self):

    with test_utils.GraphContextOrNone() as g:
      model, _, _, _ = self._create_model()

    def graph_fn():
      size = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      batch = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      shape = tf.stack([batch, size, size, 3])
      image = tf.random.uniform(shape)
      preprocessed_inputs, true_image_shapes = model.preprocess(
          image)
      prediction_dict = model.predict(preprocessed_inputs,
                                      true_image_shapes)
      detections = model.postprocess(prediction_dict, true_image_shapes)
      return [
          batch, detections['detection_boxes'], detections['detection_scores'],
          detections['detection_classes'],
          detections['detection_multiclass_scores'],
          detections['num_detections'], detections['raw_detection_boxes'],
          detections['raw_detection_scores'],
          detections['detection_anchor_indices']
      ]

    expected_boxes = [
        [
            [0, 0, .5, .5],
            [0, .5, .5, 1],
            [.5, 0, 1, .5],
            [0, 0, 0, 0],  # pruned prediction
            [0, 0, 0, 0]
        ],  # padding
        [
            [0, 0, .5, .5],
            [0, .5, .5, 1],
            [.5, 0, 1, .5],
            [0, 0, 0, 0],  # pruned prediction
            [0, 0, 0, 0]
        ]
    ]  # padding
    expected_scores = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_multiclass_scores = [[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                                  [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]

    expected_classes = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    expected_num_detections = np.array([3, 3])

    expected_raw_detection_boxes = [[[0., 0., 0.5, 0.5], [0., 0.5, 0.5, 1.],
                                     [0.5, 0., 1., 0.5], [1., 1., 1.5, 1.5]],
                                    [[0., 0., 0.5, 0.5], [0., 0.5, 0.5, 1.],
                                     [0.5, 0., 1., 0.5], [1., 1., 1.5, 1.5]]]
    expected_raw_detection_scores = [[[0, 0], [0, 0], [0, 0], [0, 0]],
                                     [[0, 0], [0, 0], [0, 0], [0, 0]]]
    expected_detection_anchor_indices = [[0, 1, 2], [0, 1, 2]]
    (batch, detection_boxes, detection_scores, detection_classes,
     detection_multiclass_scores, num_detections, raw_detection_boxes,
     raw_detection_scores, detection_anchor_indices) = self.execute_cpu(
         graph_fn, [], graph=g)
    for image_idx in range(batch):
      self.assertTrue(
          test_utils.first_rows_close_as_set(
              detection_boxes[image_idx].tolist(), expected_boxes[image_idx]))
      self.assertSameElements(detection_anchor_indices[image_idx],
                              expected_detection_anchor_indices[image_idx])
    self.assertAllClose(detection_scores, expected_scores)
    self.assertAllClose(detection_classes, expected_classes)
    self.assertAllClose(detection_multiclass_scores, expected_multiclass_scores)
    self.assertAllClose(num_detections, expected_num_detections)
    self.assertAllEqual(raw_detection_boxes, expected_raw_detection_boxes)
    self.assertAllEqual(raw_detection_scores,
                        expected_raw_detection_scores)

  def test_postprocess_results_are_correct_static(self):
    with test_utils.GraphContextOrNone() as g:
      model, _, _, _ = self._create_model(use_static_shapes=True,
                                          nms_max_size_per_class=4)

    def graph_fn(input_image):
      preprocessed_inputs, true_image_shapes = model.preprocess(input_image)
      prediction_dict = model.predict(preprocessed_inputs,
                                      true_image_shapes)
      detections = model.postprocess(prediction_dict, true_image_shapes)
      return (detections['detection_boxes'], detections['detection_scores'],
              detections['detection_classes'], detections['num_detections'],
              detections['detection_multiclass_scores'])

    expected_boxes = [
        [
            [0, 0, .5, .5],
            [0, .5, .5, 1],
            [.5, 0, 1, .5],
            [0, 0, 0, 0]
        ],  # padding
        [
            [0, 0, .5, .5],
            [0, .5, .5, 1],
            [.5, 0, 1, .5],
            [0, 0, 0, 0]
        ]
    ]  # padding
    expected_scores = [[0, 0, 0, 0], [0, 0, 0, 0]]
    expected_multiclass_scores = [[[0, 0], [0, 0], [0, 0], [0, 0]],
                                  [[0, 0], [0, 0], [0, 0], [0, 0]]]
    expected_classes = [[0, 0, 0, 0], [0, 0, 0, 0]]
    expected_num_detections = np.array([3, 3])
    batch_size = 2
    image_size = 2
    channels = 3
    input_image = np.random.rand(batch_size, image_size, image_size,
                                 channels).astype(np.float32)
    (detection_boxes, detection_scores, detection_classes,
     num_detections, detection_multiclass_scores) = self.execute(graph_fn,
                                                                 [input_image],
                                                                 graph=g)
    for image_idx in range(batch_size):
      self.assertTrue(test_utils.first_rows_close_as_set(
          detection_boxes[image_idx][
              0:expected_num_detections[image_idx]].tolist(),
          expected_boxes[image_idx][0:expected_num_detections[image_idx]]))
      self.assertAllClose(
          detection_scores[image_idx][0:expected_num_detections[image_idx]],
          expected_scores[image_idx][0:expected_num_detections[image_idx]])
      self.assertAllClose(
          detection_multiclass_scores[image_idx]
          [0:expected_num_detections[image_idx]],
          expected_multiclass_scores[image_idx]
          [0:expected_num_detections[image_idx]])
      self.assertAllClose(
          detection_classes[image_idx][0:expected_num_detections[image_idx]],
          expected_classes[image_idx][0:expected_num_detections[image_idx]])
    self.assertAllClose(num_detections,
                        expected_num_detections)

  def test_postprocess_results_are_correct_with_calibration(self):
    with test_utils.GraphContextOrNone() as g:
      model, _, _, _ = self._create_model(calibration_mapping_value=0.5)

    def graph_fn():
      size = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      batch = tf.random.uniform([], minval=2, maxval=3, dtype=tf.int32)
      shape = tf.stack([batch, size, size, 3])
      image = tf.random.uniform(shape)
      preprocessed_inputs, true_image_shapes = model.preprocess(
          image)
      prediction_dict = model.predict(preprocessed_inputs,
                                      true_image_shapes)
      detections = model.postprocess(prediction_dict, true_image_shapes)
      return detections['detection_scores'], detections['raw_detection_scores']
    # Calibration mapping value below is set to map all scores to 0.5, except
    # for the last two detections in each batch (see expected number of
    # detections below.
    expected_scores = [[0.5, 0.5, 0.5, 0., 0.], [0.5, 0.5, 0.5, 0., 0.]]
    expected_raw_detection_scores = [
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    ]
    detection_scores, raw_detection_scores = self.execute_cpu(graph_fn, [],
                                                              graph=g)
    self.assertAllClose(detection_scores, expected_scores)
    self.assertAllEqual(raw_detection_scores, expected_raw_detection_scores)

  def test_loss_results_are_correct(self):

    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, _ = self._create_model(
          apply_hard_mining=False)
    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (self._get_value_for_matching_key(loss_dict,
                                               'Loss/localization_loss'),
              self._get_value_for_matching_key(loss_dict,
                                               'Loss/classification_loss'))
    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    (localization_loss, classification_loss) = self.execute(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ],
        graph=g)

    expected_localization_loss = 0.0
    expected_classification_loss = (batch_size * num_anchors
                                    * (num_classes+1) * np.log(2.0))

    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_loss_results_are_correct_with_normalize_by_codesize_true(self):
    with test_utils.GraphContextOrNone() as g:
      model, _, _, _ = self._create_model(
          apply_hard_mining=False, normalize_loc_loss_by_codesize=True)

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (self._get_value_for_matching_key(loss_dict,
                                               'Loss/localization_loss'),)

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, 1, 1]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, 1, 1]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.5 / 4
    localization_loss = self.execute(graph_fn, [preprocessed_input,
                                                groundtruth_boxes1,
                                                groundtruth_boxes2,
                                                groundtruth_classes1,
                                                groundtruth_classes2], graph=g)
    self.assertAllClose(localization_loss, expected_localization_loss)

  def test_loss_results_are_correct_with_hard_example_mining(self):
    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, _ = self._create_model()
    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (self._get_value_for_matching_key(loss_dict,
                                               'Loss/localization_loss'),
              self._get_value_for_matching_key(loss_dict,
                                               'Loss/classification_loss'))

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    expected_classification_loss = (batch_size * num_anchors
                                    * (num_classes+1) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute_cpu(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ], graph=g)
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_loss_results_are_correct_without_add_background_class(self):

    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, _ = self._create_model(
          apply_hard_mining=False, add_background_class=False)

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(
          preprocessed_tensor, true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (loss_dict['Loss/localization_loss'],
              loss_dict['Loss/classification_loss'])

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    expected_classification_loss = (
        batch_size * num_anchors * num_classes * np.log(2.0))
    (localization_loss, classification_loss) = self.execute(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ], graph=g)

    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)


  def test_loss_results_are_correct_with_losses_mask(self):
    with test_utils.GraphContextOrNone() as g:
      model, num_classes, num_anchors, _ = self._create_model(
          apply_hard_mining=False)
    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_boxes3, groundtruth_classes1, groundtruth_classes2,
                 groundtruth_classes3):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2,
                                groundtruth_boxes3]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2,
                                  groundtruth_classes3]
      is_annotated_list = [tf.constant(True), tf.constant(True),
                           tf.constant(False)]
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list,
                                is_annotated_list=is_annotated_list)
      prediction_dict = model.predict(preprocessed_tensor,
                                      true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (self._get_value_for_matching_key(loss_dict,
                                               'Loss/localization_loss'),
              self._get_value_for_matching_key(loss_dict,
                                               'Loss/classification_loss'))

    batch_size = 3
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes3 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    groundtruth_classes3 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    # Note that we are subtracting 1 from batch_size, since the final image is
    # not annotated.
    expected_classification_loss = ((batch_size - 1) * num_anchors
                                    * (num_classes+1) * np.log(2.0))
    (localization_loss,
     classification_loss) = self.execute(graph_fn, [preprocessed_input,
                                                    groundtruth_boxes1,
                                                    groundtruth_boxes2,
                                                    groundtruth_boxes3,
                                                    groundtruth_classes1,
                                                    groundtruth_classes2,
                                                    groundtruth_classes3],
                                         graph=g)
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)

  def test_restore_map_for_detection_ckpt(self):
    # TODO(rathodv): Support TF2.X
    if self.is_tf2(): return
    model, _, _, _ = self._create_model()
    model.predict(tf.constant(np.array([[[[0, 0], [1, 1]], [[1, 0], [0, 1]]]],
                                       dtype=np.float32)),
                  true_image_shapes=None)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    save_path = self.get_temp_dir()
    with self.session() as sess:
      sess.run(init_op)
      saved_model_path = saver.save(sess, save_path)
      var_map = model.restore_map(
          fine_tune_checkpoint_type='detection',
          load_all_detection_checkpoint_vars=False)
      self.assertIsInstance(var_map, dict)
      saver = tf.train.Saver(var_map)
      saver.restore(sess, saved_model_path)
      for var in sess.run(tf.report_uninitialized_variables()):
        self.assertNotIn('FeatureExtractor', var)

  def test_restore_map_for_classification_ckpt(self):
    # TODO(rathodv): Support TF2.X
    if self.is_tf2(): return
    # Define mock tensorflow classification graph and save variables.
    test_graph_classification = tf.Graph()
    with test_graph_classification.as_default():
      image = tf.placeholder(dtype=tf.float32, shape=[1, 20, 20, 3])

      with tf.variable_scope('mock_model'):
        net = slim.conv2d(image, num_outputs=32, kernel_size=1, scope='layer1')
        slim.conv2d(net, num_outputs=3, kernel_size=1, scope='layer2')

      init_op = tf.global_variables_initializer()
      saver = tf.train.Saver()
      save_path = self.get_temp_dir()
      with self.session(graph=test_graph_classification) as sess:
        sess.run(init_op)
        saved_model_path = saver.save(sess, save_path)

    # Create tensorflow detection graph and load variables from
    # classification checkpoint.
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      model, _, _, _ = self._create_model()
      inputs_shape = [2, 2, 2, 3]
      inputs = tf.cast(tf.random_uniform(
          inputs_shape, minval=0, maxval=255, dtype=tf.int32), dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      var_map = model.restore_map(fine_tune_checkpoint_type='classification')
      self.assertNotIn('another_variable', var_map)
      self.assertIsInstance(var_map, dict)
      saver = tf.train.Saver(var_map)
      with self.session(graph=test_graph_detection) as sess:
        saver.restore(sess, saved_model_path)
        for var in sess.run(tf.report_uninitialized_variables()):
          self.assertNotIn(six.ensure_binary('FeatureExtractor'), var)

  def test_load_all_det_checkpoint_vars(self):
    if self.is_tf2(): return
    test_graph_detection = tf.Graph()
    with test_graph_detection.as_default():
      model, _, _, _ = self._create_model()
      inputs_shape = [2, 2, 2, 3]
      inputs = tf.cast(
          tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32),
          dtype=tf.float32)
      preprocessed_inputs, true_image_shapes = model.preprocess(inputs)
      prediction_dict = model.predict(preprocessed_inputs, true_image_shapes)
      model.postprocess(prediction_dict, true_image_shapes)
      another_variable = tf.Variable([17.0], name='another_variable')  # pylint: disable=unused-variable
      var_map = model.restore_map(
          fine_tune_checkpoint_type='detection',
          load_all_detection_checkpoint_vars=True)
      self.assertIsInstance(var_map, dict)
      self.assertIn('another_variable', var_map)

  def test_load_checkpoint_vars_tf2(self):

    if not self.is_tf2():
      self.skipTest('Not running TF2 checkpoint test with TF1.')

    model, _, _, _ = self._create_model()
    inputs_shape = [2, 2, 2, 3]
    inputs = tf.cast(
        tf.random_uniform(inputs_shape, minval=0, maxval=255, dtype=tf.int32),
        dtype=tf.float32)
    model(inputs)

    detection_var_names = sorted([
        var.name for var in model.restore_from_objects('detection')[
            'model']._feature_extractor.weights
    ])
    expected_detection_names = [
        'ssd_meta_arch/fake_ssd_keras_feature_extractor/mock_model/layer1/bias:0',
        'ssd_meta_arch/fake_ssd_keras_feature_extractor/mock_model/layer1/kernel:0'
    ]
    self.assertEqual(detection_var_names, expected_detection_names)

    full_var_names = sorted([
        var.name for var in
        model.restore_from_objects('full')['model'].weights
    ])

    exepcted_full_names = ['box_predictor_var:0'] + expected_detection_names
    self.assertEqual(exepcted_full_names, full_var_names)
    # TODO(vighneshb) Add similar test for classification checkpoint type.
    # TODO(vighneshb) Test loading a checkpoint from disk to verify that
    # checkpoints are loaded correctly.

  def test_loss_results_are_correct_with_random_example_sampling(self):
    with test_utils.GraphContextOrNone() as g:
      model, num_classes, _, _ = self._create_model(
          random_example_sampling=True)

    def graph_fn(preprocessed_tensor, groundtruth_boxes1, groundtruth_boxes2,
                 groundtruth_classes1, groundtruth_classes2):
      groundtruth_boxes_list = [groundtruth_boxes1, groundtruth_boxes2]
      groundtruth_classes_list = [groundtruth_classes1, groundtruth_classes2]
      model.provide_groundtruth(groundtruth_boxes_list,
                                groundtruth_classes_list)
      prediction_dict = model.predict(
          preprocessed_tensor, true_image_shapes=None)
      loss_dict = model.loss(prediction_dict, true_image_shapes=None)
      return (self._get_value_for_matching_key(loss_dict,
                                               'Loss/localization_loss'),
              self._get_value_for_matching_key(loss_dict,
                                               'Loss/classification_loss'))

    batch_size = 2
    preprocessed_input = np.random.rand(batch_size, 2, 2, 3).astype(np.float32)
    groundtruth_boxes1 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_boxes2 = np.array([[0, 0, .5, .5]], dtype=np.float32)
    groundtruth_classes1 = np.array([[1]], dtype=np.float32)
    groundtruth_classes2 = np.array([[1]], dtype=np.float32)
    expected_localization_loss = 0.0
    # Among 4 anchors (1 positive, 3 negative) in this test, only 2 anchors are
    # selected (1 positive, 1 negative) since random sampler will adjust number
    # of negative examples to make sure positive example fraction in the batch
    # is 0.5.
    expected_classification_loss = (
        batch_size * 2 * (num_classes + 1) * np.log(2.0))
    (localization_loss, classification_loss) = self.execute_cpu(
        graph_fn, [
            preprocessed_input, groundtruth_boxes1, groundtruth_boxes2,
            groundtruth_classes1, groundtruth_classes2
        ], graph=g)
    self.assertAllClose(localization_loss, expected_localization_loss)
    self.assertAllClose(classification_loss, expected_classification_loss)



if __name__ == '__main__':
  tf.test.main()
