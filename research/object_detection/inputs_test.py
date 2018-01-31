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
"""Tests for object_detection.tflearn.inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from object_detection import inputs
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util

FLAGS = tf.flags.FLAGS


def _get_configs_for_model(model_name):
  """Returns configurations for model."""
  # TODO: Make sure these tests work fine outside google3.
  fname = os.path.join(
      FLAGS.test_srcdir,
      ('google3/third_party/tensorflow_models/'
       'object_detection/samples/configs/' + model_name + '.config'))
  label_map_path = os.path.join(FLAGS.test_srcdir,
                                ('google3/third_party/tensorflow_models/'
                                 'object_detection/data/pet_label_map.pbtxt'))
  data_path = os.path.join(FLAGS.test_srcdir,
                           ('google3/third_party/tensorflow_models/'
                            'object_detection/test_data/pets_examples.record'))
  configs = config_util.get_configs_from_pipeline_file(fname)
  return config_util.merge_external_params_with_configs(
      configs,
      train_input_path=data_path,
      eval_input_path=data_path,
      label_map_path=label_map_path)


class InputsTest(tf.test.TestCase):

  def _assert_training_inputs(self, features, labels, num_classes, batch_size):
    self.assertEqual(batch_size, len(features['images']))
    self.assertEqual(batch_size, len(features['key']))
    self.assertEqual(batch_size, len(labels['locations_list']))
    self.assertEqual(batch_size, len(labels['classes_list']))
    for i in range(batch_size):
      image = features['images'][i]
      key = features['key'][i]
      locations_list = labels['locations_list'][i]
      classes_list = labels['classes_list'][i]
      weights_list = labels[fields.InputDataFields.groundtruth_weights][i]
      self.assertEqual([1, None, None, 3], image.shape.as_list())
      self.assertEqual(tf.float32, image.dtype)
      self.assertEqual(tf.string, key.dtype)
      self.assertEqual([None, 4], locations_list.shape.as_list())
      self.assertEqual(tf.float32, locations_list.dtype)
      self.assertEqual([None, num_classes], classes_list.shape.as_list())
      self.assertEqual(tf.float32, classes_list.dtype)
      self.assertEqual([None], weights_list.shape.as_list())
      self.assertEqual(tf.float32, weights_list.dtype)

  def _assert_eval_inputs(self, features, labels, num_classes):
    self.assertEqual(1, len(labels['locations_list']))
    self.assertEqual(1, len(labels['classes_list']))
    self.assertEqual(1, len(labels['image_id_list']))
    self.assertEqual(1, len(labels['area_list']))
    self.assertEqual(1, len(labels['is_crowd_list']))
    self.assertEqual(1, len(labels['difficult_list']))
    image = features['images']
    key = features['key']
    locations_list = labels['locations_list'][0]
    classes_list = labels['classes_list'][0]
    image_id_list = labels['image_id_list'][0]
    area_list = labels['area_list'][0]
    is_crowd_list = labels['is_crowd_list'][0]
    difficult_list = labels['difficult_list'][0]
    self.assertEqual([1, None, None, 3], image.shape.as_list())
    self.assertEqual(tf.float32, image.dtype)
    self.assertEqual(tf.string, key.dtype)
    self.assertEqual([None, 4], locations_list.shape.as_list())
    self.assertEqual(tf.float32, locations_list.dtype)
    self.assertEqual([None, num_classes], classes_list.shape.as_list())
    self.assertEqual(tf.float32, classes_list.dtype)
    self.assertEqual(tf.string, image_id_list.dtype)
    self.assertEqual(tf.float32, area_list.dtype)
    self.assertEqual(tf.bool, is_crowd_list.dtype)
    self.assertEqual(tf.int64, difficult_list.dtype)

  def test_faster_rcnn_resnet50_train_input(self):
    """Tests the training input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    classes = 37
    batch_size = configs['train_config'].batch_size
    train_input_fn = inputs.create_train_input_fn(
        classes, configs['train_config'], configs['train_input_config'])
    features, labels = train_input_fn()
    self._assert_training_inputs(features, labels, classes, batch_size)

  def test_faster_rcnn_resnet50_eval_input(self):
    """Tests the eval input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    classes = 37
    eval_input_fn = inputs.create_eval_input_fn(classes, configs['eval_config'],
                                                configs['eval_input_config'])
    features, labels = eval_input_fn()
    self._assert_eval_inputs(features, labels, classes)

  def test_ssd_inceptionV2_train_input(self):
    """Tests the training input function for SSDInceptionV2."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    classes = 37
    batch_size = configs['train_config'].batch_size
    train_input_fn = inputs.create_train_input_fn(
        classes, configs['train_config'], configs['train_input_config'])
    features, labels = train_input_fn()
    self._assert_training_inputs(features, labels, classes, batch_size)

  def test_ssd_inceptionV2_eval_input(self):
    """Tests the eval input function for SSDInceptionV2."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    classes = 37
    eval_input_fn = inputs.create_eval_input_fn(classes, configs['eval_config'],
                                                configs['eval_input_config'])
    features, labels = eval_input_fn()
    self._assert_eval_inputs(features, labels, classes)

  def test_predict_input(self):
    """Tests the predict input function."""
    predict_input_fn = inputs.create_predict_input_fn()
    serving_input_receiver = predict_input_fn()

    image = serving_input_receiver.features['images']
    receiver_tensors = serving_input_receiver.receiver_tensors[
        'serialized_example']
    self.assertEqual([1, None, None, 3], image.shape.as_list())
    self.assertEqual(tf.float32, image.dtype)
    self.assertEqual(tf.string, receiver_tensors.dtype)

  def test_error_with_bad_train_config(self):
    """Tests that a TypeError is raised with improper train config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    classes = 37
    train_input_fn = inputs.create_train_input_fn(
        num_classes=classes,
        train_config=configs['eval_config'],  # Expecting `TrainConfig`.
        train_input_config=configs['train_input_config'])
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_train_input_config(self):
    """Tests that a TypeError is raised with improper train input config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    classes = 37
    train_input_fn = inputs.create_train_input_fn(
        num_classes=classes,
        train_config=configs['train_config'],
        train_input_config=configs['model'])  # Expecting `InputReader`.
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_eval_config(self):
    """Tests that a TypeError is raised with improper eval config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        num_classes=classes,
        eval_config=configs['train_config'],  # Expecting `EvalConfig`.
        eval_input_config=configs['eval_input_config'])
    with self.assertRaises(TypeError):
      eval_input_fn()

  def test_error_with_bad_eval_input_config(self):
    """Tests that a TypeError is raised with improper eval input config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        num_classes=classes,
        eval_config=configs['eval_config'],
        eval_input_config=configs['model'])  # Expecting `InputReader`.
    with self.assertRaises(TypeError):
      eval_input_fn()


if __name__ == '__main__':
  tf.test.main()
