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
"""Tests for object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import numpy as np
import tensorflow as tf

from object_detection import inputs
from object_detection import model
from object_detection import model_hparams
from object_detection import model_test_util
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util

FLAGS = tf.flags.FLAGS

MODEL_NAME_FOR_TEST = model_test_util.SSD_INCEPTION_MODEL_NAME


def _get_data_path():
  """Returns an absolute path to TFRecord file."""
  return os.path.join(FLAGS.test_srcdir, model_test_util.PATH_BASE, 'test_data',
                      'pets_examples.record')


def _get_labelmap_path():
  """Returns an absolute path to label map file."""
  return os.path.join(FLAGS.test_srcdir, model_test_util.PATH_BASE, 'data',
                      'pet_label_map.pbtxt')


def _get_configs_for_model(model_name):
  """Returns configurations for model."""
  filename = model_test_util.GetPipelineConfigPath(model_name)
  data_path = _get_data_path()
  label_map_path = _get_labelmap_path()
  configs = config_util.get_configs_from_pipeline_file(filename)
  configs = config_util.merge_external_params_with_configs(
      configs,
      train_input_path=data_path,
      eval_input_path=data_path,
      label_map_path=label_map_path)
  return configs


def setUpModule():
  model_test_util.InitializeFlags(MODEL_NAME_FOR_TEST)


class ModelTflearnTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.reset_default_graph()

  def _assert_outputs_for_train_eval(self, configs, mode, class_agnostic=False):
    model_config = configs['model']
    train_config = configs['train_config']
    with tf.Graph().as_default():
      if mode == tf.estimator.ModeKeys.TRAIN:
        features, labels = inputs.create_train_input_fn(
            configs['train_config'],
            configs['train_input_config'],
            configs['model'])()
        batch_size = train_config.batch_size
      else:
        features, labels = inputs.create_eval_input_fn(
            configs['eval_config'],
            configs['eval_input_config'],
            configs['model'])()
        batch_size = 1

      detection_model_fn = functools.partial(
          model_builder.build, model_config=model_config, is_training=True)

      hparams = model_hparams.create_hparams(
          hparams_overrides='load_pretrained=false')

      model_fn = model.create_model_fn(detection_model_fn, configs, hparams)
      estimator_spec = model_fn(features, labels, mode)

      self.assertIsNotNone(estimator_spec.loss)
      self.assertIsNotNone(estimator_spec.predictions)
      if class_agnostic:
        self.assertNotIn('detection_classes', estimator_spec.predictions)
      else:
        detection_classes = estimator_spec.predictions['detection_classes']
        self.assertEqual(batch_size, detection_classes.shape.as_list()[0])
        self.assertEqual(tf.float32, detection_classes.dtype)
      detection_boxes = estimator_spec.predictions['detection_boxes']
      detection_scores = estimator_spec.predictions['detection_scores']
      num_detections = estimator_spec.predictions['num_detections']
      self.assertEqual(batch_size, detection_boxes.shape.as_list()[0])
      self.assertEqual(tf.float32, detection_boxes.dtype)
      self.assertEqual(batch_size, detection_scores.shape.as_list()[0])
      self.assertEqual(tf.float32, detection_scores.dtype)
      self.assertEqual(tf.float32, num_detections.dtype)
      if mode == tf.estimator.ModeKeys.TRAIN:
        self.assertIsNotNone(estimator_spec.train_op)
      return estimator_spec

  def _assert_outputs_for_predict(self, configs):
    model_config = configs['model']

    with tf.Graph().as_default():
      features, _ = inputs.create_eval_input_fn(
          configs['eval_config'],
          configs['eval_input_config'],
          configs['model'])()
      detection_model_fn = functools.partial(
          model_builder.build, model_config=model_config, is_training=False)

      hparams = model_hparams.create_hparams(
          hparams_overrides='load_pretrained=false')

      model_fn = model.create_model_fn(detection_model_fn, configs, hparams)
      estimator_spec = model_fn(features, None, tf.estimator.ModeKeys.PREDICT)

      self.assertIsNone(estimator_spec.loss)
      self.assertIsNone(estimator_spec.train_op)
      self.assertIsNotNone(estimator_spec.predictions)
      self.assertIsNotNone(estimator_spec.export_outputs)
      self.assertIn(tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    estimator_spec.export_outputs)

  def testModelFnInTrainMode(self):
    """Tests the model function in TRAIN mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_outputs_for_train_eval(configs, tf.estimator.ModeKeys.TRAIN)

  def testModelFnInEvalMode(self):
    """Tests the model function in EVAL mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_outputs_for_train_eval(configs, tf.estimator.ModeKeys.EVAL)

  def testModelFnInPredictMode(self):
    """Tests the model function in PREDICT mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_outputs_for_predict(configs)

  def testExperiment(self):
    """Tests that the `Experiment` object is constructed correctly."""
    experiment = model_test_util.BuildExperiment()
    model_dir = experiment.estimator.model_dir
    pipeline_config_path = os.path.join(model_dir, 'pipeline.config')
    self.assertTrue(tf.gfile.Exists(pipeline_config_path))


class UnbatchTensorsTest(tf.test.TestCase):

  def test_unbatch_without_unpadding(self):
    image_placeholder = tf.placeholder(tf.float32, [2, None, None, None])
    groundtruth_boxes_placeholder = tf.placeholder(tf.float32, [2, None, None])
    groundtruth_classes_placeholder = tf.placeholder(tf.float32,
                                                     [2, None, None])
    groundtruth_weights_placeholder = tf.placeholder(tf.float32, [2, None])

    tensor_dict = {
        fields.InputDataFields.image:
            image_placeholder,
        fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes_placeholder,
        fields.InputDataFields.groundtruth_classes:
            groundtruth_classes_placeholder,
        fields.InputDataFields.groundtruth_weights:
            groundtruth_weights_placeholder
    }
    unbatched_tensor_dict = model.unstack_batch(
        tensor_dict, unpad_groundtruth_tensors=False)

    with self.test_session() as sess:
      unbatched_tensor_dict_out = sess.run(
          unbatched_tensor_dict,
          feed_dict={
              image_placeholder:
                  np.random.rand(2, 4, 4, 3).astype(np.float32),
              groundtruth_boxes_placeholder:
                  np.random.rand(2, 5, 4).astype(np.float32),
              groundtruth_classes_placeholder:
                  np.random.rand(2, 5, 6).astype(np.float32),
              groundtruth_weights_placeholder:
                  np.random.rand(2, 5).astype(np.float32)
          })
    for image_out in unbatched_tensor_dict_out[fields.InputDataFields.image]:
      self.assertAllEqual(image_out.shape, [4, 4, 3])
    for groundtruth_boxes_out in unbatched_tensor_dict_out[
        fields.InputDataFields.groundtruth_boxes]:
      self.assertAllEqual(groundtruth_boxes_out.shape, [5, 4])
    for groundtruth_classes_out in unbatched_tensor_dict_out[
        fields.InputDataFields.groundtruth_classes]:
      self.assertAllEqual(groundtruth_classes_out.shape, [5, 6])
    for groundtruth_weights_out in unbatched_tensor_dict_out[
        fields.InputDataFields.groundtruth_weights]:
      self.assertAllEqual(groundtruth_weights_out.shape, [5])

  def test_unbatch_and_unpad_groundtruth_tensors(self):
    image_placeholder = tf.placeholder(tf.float32, [2, None, None, None])
    groundtruth_boxes_placeholder = tf.placeholder(tf.float32, [2, 5, None])
    groundtruth_classes_placeholder = tf.placeholder(tf.float32, [2, 5, None])
    groundtruth_weights_placeholder = tf.placeholder(tf.float32, [2, 5])
    num_groundtruth_placeholder = tf.placeholder(tf.int32, [2])

    tensor_dict = {
        fields.InputDataFields.image:
            image_placeholder,
        fields.InputDataFields.groundtruth_boxes:
            groundtruth_boxes_placeholder,
        fields.InputDataFields.groundtruth_classes:
            groundtruth_classes_placeholder,
        fields.InputDataFields.groundtruth_weights:
            groundtruth_weights_placeholder,
        fields.InputDataFields.num_groundtruth_boxes:
            num_groundtruth_placeholder
    }
    unbatched_tensor_dict = model.unstack_batch(
        tensor_dict, unpad_groundtruth_tensors=True)
    with self.test_session() as sess:
      unbatched_tensor_dict_out = sess.run(
          unbatched_tensor_dict,
          feed_dict={
              image_placeholder:
                  np.random.rand(2, 4, 4, 3).astype(np.float32),
              groundtruth_boxes_placeholder:
                  np.random.rand(2, 5, 4).astype(np.float32),
              groundtruth_classes_placeholder:
                  np.random.rand(2, 5, 6).astype(np.float32),
              groundtruth_weights_placeholder:
                  np.random.rand(2, 5).astype(np.float32),
              num_groundtruth_placeholder:
                  np.array([3, 3], np.int32)
          })
    for image_out in unbatched_tensor_dict_out[fields.InputDataFields.image]:
      self.assertAllEqual(image_out.shape, [4, 4, 3])
    for groundtruth_boxes_out in unbatched_tensor_dict_out[
        fields.InputDataFields.groundtruth_boxes]:
      self.assertAllEqual(groundtruth_boxes_out.shape, [3, 4])
    for groundtruth_classes_out in unbatched_tensor_dict_out[
        fields.InputDataFields.groundtruth_classes]:
      self.assertAllEqual(groundtruth_classes_out.shape, [3, 6])
    for groundtruth_weights_out in unbatched_tensor_dict_out[
        fields.InputDataFields.groundtruth_weights]:
      self.assertAllEqual(groundtruth_weights_out.shape, [3])


if __name__ == '__main__':
  tf.test.main()
