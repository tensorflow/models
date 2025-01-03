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
"""Tests for object detection model library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import unittest
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from object_detection import inputs
from object_detection import model_hparams
from object_detection import model_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import tf_version


# Model for test. Options are:
# 'ssd_inception_v2_pets', 'faster_rcnn_resnet50_pets'
MODEL_NAME_FOR_TEST = 'ssd_inception_v2_pets'

# Model for testing keypoints.
MODEL_NAME_FOR_KEYPOINTS_TEST = 'ssd_mobilenet_v1_fpp'

# Model for testing tfSequenceExample inputs.
MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST = 'context_rcnn_camera_trap'


def _get_data_path(model_name):
  """Returns an absolute path to TFRecord file."""
  if model_name == MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST:
    return os.path.join(tf.resource_loader.get_data_files_path(), 'test_data',
                        'snapshot_serengeti_sequence_examples.record')
  else:
    return os.path.join(tf.resource_loader.get_data_files_path(), 'test_data',
                        'pets_examples.record')


def get_pipeline_config_path(model_name):
  """Returns path to the local pipeline config file."""
  if model_name == MODEL_NAME_FOR_KEYPOINTS_TEST:
    return os.path.join(tf.resource_loader.get_data_files_path(), 'test_data',
                        model_name + '.config')
  elif model_name == MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST:
    return os.path.join(tf.resource_loader.get_data_files_path(), 'test_data',
                        model_name + '.config')
  else:
    return os.path.join(tf.resource_loader.get_data_files_path(), 'samples',
                        'configs', model_name + '.config')


def _get_labelmap_path():
  """Returns an absolute path to label map file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'data',
                      'pet_label_map.pbtxt')


def _get_keypoints_labelmap_path():
  """Returns an absolute path to label map file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'data',
                      'face_person_with_keypoints_label_map.pbtxt')


def _get_sequence_example_labelmap_path():
  """Returns an absolute path to label map file."""
  return os.path.join(tf.resource_loader.get_data_files_path(), 'data',
                      'snapshot_serengeti_label_map.pbtxt')


def _get_configs_for_model(model_name):
  """Returns configurations for model."""
  filename = get_pipeline_config_path(model_name)
  data_path = _get_data_path(model_name)
  if model_name == MODEL_NAME_FOR_KEYPOINTS_TEST:
    label_map_path = _get_keypoints_labelmap_path()
  elif model_name == MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST:
    label_map_path = _get_sequence_example_labelmap_path()
  else:
    label_map_path = _get_labelmap_path()
  configs = config_util.get_configs_from_pipeline_file(filename)
  override_dict = {
      'train_input_path': data_path,
      'eval_input_path': data_path,
      'label_map_path': label_map_path
  }
  configs = config_util.merge_external_params_with_configs(
      configs, kwargs_dict=override_dict)
  return configs


def _make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = tf.data.make_initializable_iterator(dataset)
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ModelLibTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.reset_default_graph()

  def _assert_model_fn_for_train_eval(self, configs, mode,
                                      class_agnostic=False):
    model_config = configs['model']
    train_config = configs['train_config']
    with tf.Graph().as_default():
      if mode == 'train':
        features, labels = _make_initializable_iterator(
            inputs.create_train_input_fn(configs['train_config'],
                                         configs['train_input_config'],
                                         configs['model'])()).get_next()
        model_mode = tf_estimator.ModeKeys.TRAIN
        batch_size = train_config.batch_size
      elif mode == 'eval':
        features, labels = _make_initializable_iterator(
            inputs.create_eval_input_fn(configs['eval_config'],
                                        configs['eval_input_config'],
                                        configs['model'])()).get_next()
        model_mode = tf_estimator.ModeKeys.EVAL
        batch_size = 1
      elif mode == 'eval_on_train':
        features, labels = _make_initializable_iterator(
            inputs.create_eval_input_fn(configs['eval_config'],
                                        configs['train_input_config'],
                                        configs['model'])()).get_next()
        model_mode = tf_estimator.ModeKeys.EVAL
        batch_size = 1

      detection_model_fn = functools.partial(
          model_builder.build, model_config=model_config, is_training=True)

      hparams = model_hparams.create_hparams(
          hparams_overrides='load_pretrained=false')

      model_fn = model_lib.create_model_fn(detection_model_fn, configs, hparams)
      estimator_spec = model_fn(features, labels, model_mode)

      self.assertIsNotNone(estimator_spec.loss)
      self.assertIsNotNone(estimator_spec.predictions)
      if mode == 'eval' or mode == 'eval_on_train':
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
        if mode == 'eval':
          self.assertIn('Detections_Left_Groundtruth_Right/0',
                        estimator_spec.eval_metric_ops)
      if model_mode == tf_estimator.ModeKeys.TRAIN:
        self.assertIsNotNone(estimator_spec.train_op)
      return estimator_spec

  def _assert_model_fn_for_predict(self, configs):
    model_config = configs['model']

    with tf.Graph().as_default():
      features, _ = _make_initializable_iterator(
          inputs.create_eval_input_fn(configs['eval_config'],
                                      configs['eval_input_config'],
                                      configs['model'])()).get_next()
      detection_model_fn = functools.partial(
          model_builder.build, model_config=model_config, is_training=False)

      hparams = model_hparams.create_hparams(
          hparams_overrides='load_pretrained=false')

      model_fn = model_lib.create_model_fn(detection_model_fn, configs, hparams)
      estimator_spec = model_fn(features, None, tf_estimator.ModeKeys.PREDICT)

      self.assertIsNone(estimator_spec.loss)
      self.assertIsNone(estimator_spec.train_op)
      self.assertIsNotNone(estimator_spec.predictions)
      self.assertIsNotNone(estimator_spec.export_outputs)
      self.assertIn(tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                    estimator_spec.export_outputs)

  def test_model_fn_in_train_mode(self):
    """Tests the model function in TRAIN mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_model_fn_for_train_eval(configs, 'train')

  def test_model_fn_in_train_mode_sequences(self):
    """Tests the model function in TRAIN mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST)
    self._assert_model_fn_for_train_eval(configs, 'train')

  def test_model_fn_in_train_mode_freeze_all_variables(self):
    """Tests model_fn TRAIN mode with all variables frozen."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    configs['train_config'].freeze_variables.append('.*')
    with self.assertRaisesRegexp(ValueError, 'No variables to optimize'):
      self._assert_model_fn_for_train_eval(configs, 'train')

  def test_model_fn_in_train_mode_freeze_all_included_variables(self):
    """Tests model_fn TRAIN mode with all included variables frozen."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    train_config = configs['train_config']
    train_config.update_trainable_variables.append('FeatureExtractor')
    train_config.freeze_variables.append('.*')
    with self.assertRaisesRegexp(ValueError, 'No variables to optimize'):
      self._assert_model_fn_for_train_eval(configs, 'train')

  def test_model_fn_in_train_mode_freeze_box_predictor(self):
    """Tests model_fn TRAIN mode with FeatureExtractor variables frozen."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    train_config = configs['train_config']
    train_config.update_trainable_variables.append('FeatureExtractor')
    train_config.update_trainable_variables.append('BoxPredictor')
    train_config.freeze_variables.append('FeatureExtractor')
    self._assert_model_fn_for_train_eval(configs, 'train')

  def test_model_fn_in_eval_mode(self):
    """Tests the model function in EVAL mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_model_fn_for_train_eval(configs, 'eval')

  def test_model_fn_in_eval_mode_sequences(self):
    """Tests the model function in EVAL mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST)
    self._assert_model_fn_for_train_eval(configs, 'eval')

  def test_model_fn_in_keypoints_eval_mode(self):
    """Tests the model function in EVAL mode with keypoints config."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_KEYPOINTS_TEST)
    estimator_spec = self._assert_model_fn_for_train_eval(configs, 'eval')
    metric_ops = estimator_spec.eval_metric_ops
    self.assertIn('Keypoints_Precision/mAP ByCategory/face', metric_ops)
    self.assertIn('Keypoints_Precision/mAP ByCategory/PERSON', metric_ops)
    detection_keypoints = estimator_spec.predictions['detection_keypoints']
    self.assertEqual(1, detection_keypoints.shape.as_list()[0])
    self.assertEqual(tf.float32, detection_keypoints.dtype)

  def test_model_fn_in_eval_on_train_mode(self):
    """Tests the model function in EVAL mode with train data."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_model_fn_for_train_eval(configs, 'eval_on_train')

  def test_model_fn_in_predict_mode(self):
    """Tests the model function in PREDICT mode."""
    configs = _get_configs_for_model(MODEL_NAME_FOR_TEST)
    self._assert_model_fn_for_predict(configs)

  def test_create_estimator_and_inputs(self):
    """Tests that Estimator and input function are constructed correctly."""
    run_config = tf_estimator.RunConfig()
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    train_steps = 20
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config,
        hparams,
        pipeline_config_path,
        train_steps=train_steps)
    estimator = train_and_eval_dict['estimator']
    train_steps = train_and_eval_dict['train_steps']
    self.assertIsInstance(estimator, tf_estimator.Estimator)
    self.assertEqual(20, train_steps)
    self.assertIn('train_input_fn', train_and_eval_dict)
    self.assertIn('eval_input_fns', train_and_eval_dict)
    self.assertIn('eval_on_train_input_fn', train_and_eval_dict)

  def test_create_estimator_and_inputs_sequence_example(self):
    """Tests that Estimator and input function are constructed correctly."""
    run_config = tf_estimator.RunConfig()
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(
        MODEL_NAME_FOR_SEQUENCE_EXAMPLE_TEST)
    train_steps = 20
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config,
        hparams,
        pipeline_config_path,
        train_steps=train_steps)
    estimator = train_and_eval_dict['estimator']
    train_steps = train_and_eval_dict['train_steps']
    self.assertIsInstance(estimator, tf_estimator.Estimator)
    self.assertEqual(20, train_steps)
    self.assertIn('train_input_fn', train_and_eval_dict)
    self.assertIn('eval_input_fns', train_and_eval_dict)
    self.assertIn('eval_on_train_input_fn', train_and_eval_dict)

  def test_create_estimator_with_default_train_eval_steps(self):
    """Tests that number of train/eval defaults to config values."""
    run_config = tf_estimator.RunConfig()
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    config_train_steps = configs['train_config'].num_steps
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config, hparams, pipeline_config_path)
    estimator = train_and_eval_dict['estimator']
    train_steps = train_and_eval_dict['train_steps']

    self.assertIsInstance(estimator, tf_estimator.Estimator)
    self.assertEqual(config_train_steps, train_steps)

  def test_create_tpu_estimator_and_inputs(self):
    """Tests that number of train/eval defaults to config values."""
    run_config = tf_estimator.tpu.RunConfig()
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    train_steps = 20
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config,
        hparams,
        pipeline_config_path,
        train_steps=train_steps,
        use_tpu_estimator=True)
    estimator = train_and_eval_dict['estimator']
    train_steps = train_and_eval_dict['train_steps']

    self.assertIsInstance(estimator, tf_estimator.tpu.TPUEstimator)
    self.assertEqual(20, train_steps)

  def test_create_train_and_eval_specs(self):
    """Tests that `TrainSpec` and `EvalSpec` is created correctly."""
    run_config = tf_estimator.RunConfig()
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    train_steps = 20
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config,
        hparams,
        pipeline_config_path,
        train_steps=train_steps)
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=True,
        final_exporter_name='exporter',
        eval_spec_names=['holdout'])
    self.assertEqual(train_steps, train_spec.max_steps)
    self.assertEqual(2, len(eval_specs))
    self.assertEqual(None, eval_specs[0].steps)
    self.assertEqual('holdout', eval_specs[0].name)
    self.assertEqual('exporter', eval_specs[0].exporters[0].name)
    self.assertEqual(None, eval_specs[1].steps)
    self.assertEqual('eval_on_train', eval_specs[1].name)

  def test_experiment(self):
    """Tests that the `Experiment` object is constructed correctly."""
    run_config = tf_estimator.RunConfig()
    hparams = model_hparams.create_hparams(
        hparams_overrides='load_pretrained=false')
    pipeline_config_path = get_pipeline_config_path(MODEL_NAME_FOR_TEST)
    experiment = model_lib.populate_experiment(
        run_config,
        hparams,
        pipeline_config_path,
        train_steps=10,
        eval_steps=20)
    self.assertEqual(10, experiment.train_steps)
    self.assertEqual(None, experiment.eval_steps)


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
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
    unbatched_tensor_dict = model_lib.unstack_batch(
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
    unbatched_tensor_dict = model_lib.unstack_batch(
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
