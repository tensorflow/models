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

import functools
import os
from absl import logging
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from object_detection import inputs
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util
from object_detection.utils import test_case

FLAGS = tf.flags.FLAGS


def _get_configs_for_model(model_name):
  """Returns configurations for model."""
  fname = os.path.join(tf.resource_loader.get_data_files_path(),
                       'samples/configs/' + model_name + '.config')
  label_map_path = os.path.join(tf.resource_loader.get_data_files_path(),
                                'data/pet_label_map.pbtxt')
  data_path = os.path.join(tf.resource_loader.get_data_files_path(),
                           'test_data/pets_examples.record')
  configs = config_util.get_configs_from_pipeline_file(fname)
  override_dict = {
      'train_input_path': data_path,
      'eval_input_path': data_path,
      'label_map_path': label_map_path
  }
  return config_util.merge_external_params_with_configs(
      configs, kwargs_dict=override_dict)


def _make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.

  Args:
    dataset: A `tf.data.Dataset` object.

  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


class InputsTest(test_case.TestCase, parameterized.TestCase):

  def test_faster_rcnn_resnet50_train_input(self):
    """Tests the training input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    model_config = configs['model']
    model_config.faster_rcnn.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        configs['train_config'], configs['train_input_config'], model_config)
    features, labels = _make_initializable_iterator(train_input_fn()).get_next()

    self.assertAllEqual([1, None, None, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual([1],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [1, 100, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [1, 100, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [1, 100],
        labels[fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_weights].dtype)
    self.assertAllEqual(
        [1, 100, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_confidences].shape.as_list())
    self.assertEqual(
        tf.float32,
        labels[fields.InputDataFields.groundtruth_confidences].dtype)

  def test_faster_rcnn_resnet50_train_input_with_additional_channels(self):
    """Tests the training input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    model_config = configs['model']
    configs['train_input_config'].num_additional_channels = 2
    configs['train_config'].retain_original_images = True
    model_config.faster_rcnn.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        configs['train_config'], configs['train_input_config'], model_config)
    features, labels = _make_initializable_iterator(train_input_fn()).get_next()

    self.assertAllEqual([1, None, None, 5],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertAllEqual(
        [1, None, None, 3],
        features[fields.InputDataFields.original_image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual([1],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [1, 100, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [1, 100, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [1, 100],
        labels[fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_weights].dtype)
    self.assertAllEqual(
        [1, 100, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_confidences].shape.as_list())
    self.assertEqual(
        tf.float32,
        labels[fields.InputDataFields.groundtruth_confidences].dtype)

  @parameterized.parameters(
      {'eval_batch_size': 1},
      {'eval_batch_size': 8}
  )
  def test_faster_rcnn_resnet50_eval_input(self, eval_batch_size=1):
    """Tests the eval input function for FasterRcnnResnet50."""
    configs = _get_configs_for_model('faster_rcnn_resnet50_pets')
    model_config = configs['model']
    model_config.faster_rcnn.num_classes = 37
    eval_config = configs['eval_config']
    eval_config.batch_size = eval_batch_size
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config, configs['eval_input_configs'][0], model_config)
    features, labels = _make_initializable_iterator(eval_input_fn()).get_next()
    self.assertAllEqual([eval_batch_size, None, None, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual(
        [eval_batch_size, None, None, 3],
        features[fields.InputDataFields.original_image].shape.as_list())
    self.assertEqual(tf.uint8,
                     features[fields.InputDataFields.original_image].dtype)
    self.assertAllEqual([eval_batch_size],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100, model_config.faster_rcnn.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(
        tf.float32,
        labels[fields.InputDataFields.groundtruth_weights].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_area].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_area].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_is_crowd].shape.as_list())
    self.assertEqual(
        tf.bool, labels[fields.InputDataFields.groundtruth_is_crowd].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_difficult].shape.as_list())
    self.assertEqual(
        tf.int32, labels[fields.InputDataFields.groundtruth_difficult].dtype)

  def test_ssd_inceptionV2_train_input(self):
    """Tests the training input function for SSDInceptionV2."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    model_config = configs['model']
    model_config.ssd.num_classes = 37
    batch_size = configs['train_config'].batch_size
    train_input_fn = inputs.create_train_input_fn(
        configs['train_config'], configs['train_input_config'], model_config)
    features, labels = _make_initializable_iterator(train_input_fn()).get_next()

    self.assertAllEqual([batch_size, 300, 300, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual([batch_size],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [batch_size],
        labels[fields.InputDataFields.num_groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.int32,
                     labels[fields.InputDataFields.num_groundtruth_boxes].dtype)
    self.assertAllEqual(
        [batch_size, 100, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [batch_size, 100, model_config.ssd.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [batch_size, 100],
        labels[
            fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(
        tf.float32,
        labels[fields.InputDataFields.groundtruth_weights].dtype)

  @parameterized.parameters(
      {'eval_batch_size': 1},
      {'eval_batch_size': 8}
  )
  def test_ssd_inceptionV2_eval_input(self, eval_batch_size=1):
    """Tests the eval input function for SSDInceptionV2."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    model_config = configs['model']
    model_config.ssd.num_classes = 37
    eval_config = configs['eval_config']
    eval_config.batch_size = eval_batch_size
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config, configs['eval_input_configs'][0], model_config)
    features, labels = _make_initializable_iterator(eval_input_fn()).get_next()
    self.assertAllEqual([eval_batch_size, 300, 300, 3],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual(
        [eval_batch_size, 300, 300, 3],
        features[fields.InputDataFields.original_image].shape.as_list())
    self.assertEqual(tf.uint8,
                     features[fields.InputDataFields.original_image].dtype)
    self.assertAllEqual([eval_batch_size],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100, model_config.ssd.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[
            fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(
        tf.float32,
        labels[fields.InputDataFields.groundtruth_weights].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_area].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_area].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_is_crowd].shape.as_list())
    self.assertEqual(
        tf.bool, labels[fields.InputDataFields.groundtruth_is_crowd].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_difficult].shape.as_list())
    self.assertEqual(
        tf.int32, labels[fields.InputDataFields.groundtruth_difficult].dtype)

  def test_ssd_inceptionV2_eval_input_with_additional_channels(
      self, eval_batch_size=1):
    """Tests the eval input function for SSDInceptionV2 with additional channels.

    Args:
      eval_batch_size: Batch size for eval set.
    """
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    model_config = configs['model']
    model_config.ssd.num_classes = 37
    configs['eval_input_configs'][0].num_additional_channels = 1
    eval_config = configs['eval_config']
    eval_config.batch_size = eval_batch_size
    eval_config.retain_original_image_additional_channels = True
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config, configs['eval_input_configs'][0], model_config)
    features, labels = _make_initializable_iterator(eval_input_fn()).get_next()
    self.assertAllEqual([eval_batch_size, 300, 300, 4],
                        features[fields.InputDataFields.image].shape.as_list())
    self.assertEqual(tf.float32, features[fields.InputDataFields.image].dtype)
    self.assertAllEqual(
        [eval_batch_size, 300, 300, 3],
        features[fields.InputDataFields.original_image].shape.as_list())
    self.assertEqual(tf.uint8,
                     features[fields.InputDataFields.original_image].dtype)
    self.assertAllEqual([eval_batch_size, 300, 300, 1], features[
        fields.InputDataFields.image_additional_channels].shape.as_list())
    self.assertEqual(
        tf.uint8,
        features[fields.InputDataFields.image_additional_channels].dtype)
    self.assertAllEqual([eval_batch_size],
                        features[inputs.HASH_KEY].shape.as_list())
    self.assertEqual(tf.int32, features[inputs.HASH_KEY].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100, 4],
        labels[fields.InputDataFields.groundtruth_boxes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_boxes].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100, model_config.ssd.num_classes],
        labels[fields.InputDataFields.groundtruth_classes].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_classes].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_weights].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_weights].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_area].shape.as_list())
    self.assertEqual(tf.float32,
                     labels[fields.InputDataFields.groundtruth_area].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_is_crowd].shape.as_list())
    self.assertEqual(tf.bool,
                     labels[fields.InputDataFields.groundtruth_is_crowd].dtype)
    self.assertAllEqual(
        [eval_batch_size, 100],
        labels[fields.InputDataFields.groundtruth_difficult].shape.as_list())
    self.assertEqual(tf.int32,
                     labels[fields.InputDataFields.groundtruth_difficult].dtype)

  def test_predict_input(self):
    """Tests the predict input function."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    predict_input_fn = inputs.create_predict_input_fn(
        model_config=configs['model'],
        predict_input_config=configs['eval_input_configs'][0])
    serving_input_receiver = predict_input_fn()

    image = serving_input_receiver.features[fields.InputDataFields.image]
    receiver_tensors = serving_input_receiver.receiver_tensors[
        inputs.SERVING_FED_EXAMPLE_KEY]
    self.assertEqual([1, 300, 300, 3], image.shape.as_list())
    self.assertEqual(tf.float32, image.dtype)
    self.assertEqual(tf.string, receiver_tensors.dtype)

  def test_predict_input_with_additional_channels(self):
    """Tests the predict input function with additional channels."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['eval_input_configs'][0].num_additional_channels = 2
    predict_input_fn = inputs.create_predict_input_fn(
        model_config=configs['model'],
        predict_input_config=configs['eval_input_configs'][0])
    serving_input_receiver = predict_input_fn()

    image = serving_input_receiver.features[fields.InputDataFields.image]
    receiver_tensors = serving_input_receiver.receiver_tensors[
        inputs.SERVING_FED_EXAMPLE_KEY]
    # RGB + 2 additional channels = 5 channels.
    self.assertEqual([1, 300, 300, 5], image.shape.as_list())
    self.assertEqual(tf.float32, image.dtype)
    self.assertEqual(tf.string, receiver_tensors.dtype)

  def test_error_with_bad_train_config(self):
    """Tests that a TypeError is raised with improper train config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['eval_config'],  # Expecting `TrainConfig`.
        train_input_config=configs['train_input_config'],
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_train_input_config(self):
    """Tests that a TypeError is raised with improper train input config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['train_config'],
        train_input_config=configs['model'],  # Expecting `InputReader`.
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_train_model_config(self):
    """Tests that a TypeError is raised with improper train model config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['train_config'],
        train_input_config=configs['train_input_config'],
        model_config=configs['train_config'])  # Expecting `DetectionModel`.
    with self.assertRaises(TypeError):
      train_input_fn()

  def test_error_with_bad_eval_config(self):
    """Tests that a TypeError is raised with improper eval config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['train_config'],  # Expecting `EvalConfig`.
        eval_input_config=configs['eval_input_configs'][0],
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      eval_input_fn()

  def test_error_with_bad_eval_input_config(self):
    """Tests that a TypeError is raised with improper eval input config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['eval_config'],
        eval_input_config=configs['model'],  # Expecting `InputReader`.
        model_config=configs['model'])
    with self.assertRaises(TypeError):
      eval_input_fn()

  def test_error_with_bad_eval_model_config(self):
    """Tests that a TypeError is raised with improper eval model config."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['model'].ssd.num_classes = 37
    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['eval_config'],
        eval_input_config=configs['eval_input_configs'][0],
        model_config=configs['eval_config'])  # Expecting `DetectionModel`.
    with self.assertRaises(TypeError):
      eval_input_fn()

  def test_output_equal_in_replace_empty_string_with_random_number(self):
    string_placeholder = tf.placeholder(tf.string, shape=[])
    replaced_string = inputs._replace_empty_string_with_random_number(
        string_placeholder)

    test_string = b'hello world'
    feed_dict = {string_placeholder: test_string}

    with self.test_session() as sess:
      out_string = sess.run(replaced_string, feed_dict=feed_dict)

    self.assertEqual(test_string, out_string)

  def test_output_is_integer_in_replace_empty_string_with_random_number(self):

    string_placeholder = tf.placeholder(tf.string, shape=[])
    replaced_string = inputs._replace_empty_string_with_random_number(
        string_placeholder)

    empty_string = ''
    feed_dict = {string_placeholder: empty_string}
    with self.test_session() as sess:
      out_string = sess.run(replaced_string, feed_dict=feed_dict)

    is_integer = True
    try:
      # Test whether out_string is a string which represents an integer, the
      # casting below will throw an error if out_string is not castable to int.
      int(out_string)
    except ValueError:
      is_integer = False

    self.assertTrue(is_integer)

  def test_force_no_resize(self):
    """Tests the functionality of force_no_reisze option."""
    configs = _get_configs_for_model('ssd_inception_v2_pets')
    configs['eval_config'].force_no_resize = True

    eval_input_fn = inputs.create_eval_input_fn(
        eval_config=configs['eval_config'],
        eval_input_config=configs['eval_input_configs'][0],
        model_config=configs['model']
    )
    train_input_fn = inputs.create_train_input_fn(
        train_config=configs['train_config'],
        train_input_config=configs['train_input_config'],
        model_config=configs['model']
    )

    features_train, _ = _make_initializable_iterator(
        train_input_fn()).get_next()

    features_eval, _ = _make_initializable_iterator(
        eval_input_fn()).get_next()

    images_train, images_eval = features_train['image'], features_eval['image']

    self.assertEqual([1, None, None, 3], images_eval.shape.as_list())
    self.assertEqual([24, 300, 300, 3], images_train.shape.as_list())


class DataAugmentationFnTest(test_case.TestCase):

  def test_apply_image_and_box_augmentation(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        }),
        (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1., 1.]], np.float32))
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_boxes],
        [[10, 10, 20, 20]]
    )

  def test_apply_image_and_box_augmentation_with_scores(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        }),
        (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1., 1.]], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1.0], np.float32)),
        fields.InputDataFields.groundtruth_weights:
            tf.constant(np.array([0.8], np.float32)),
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_boxes],
        [[10, 10, 20, 20]]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_classes],
        [1.0]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[
            fields.InputDataFields.groundtruth_weights],
        [0.8]
    )

  def test_include_masks_in_data_augmentation(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        })
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_instance_masks:
            tf.constant(np.zeros([2, 10, 10], np.uint8))
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3])
    self.assertAllEqual(augmented_tensor_dict_out[
        fields.InputDataFields.groundtruth_instance_masks].shape, [2, 20, 20])

  def test_include_keypoints_in_data_augmentation(self):
    data_augmentation_options = [
        (preprocessor.resize_image, {
            'new_height': 20,
            'new_width': 20,
            'method': tf.image.ResizeMethod.NEAREST_NEIGHBOR
        }),
        (preprocessor.scale_boxes_to_pixel_coordinates, {}),
    ]
    data_augmentation_fn = functools.partial(
        inputs.augment_input_data,
        data_augmentation_options=data_augmentation_options)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(10, 10, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1., 1.]], np.float32)),
        fields.InputDataFields.groundtruth_keypoints:
            tf.constant(np.array([[[0.5, 1.0], [0.5, 0.5]]], np.float32))
    }
    augmented_tensor_dict = data_augmentation_fn(tensor_dict=tensor_dict)
    with self.test_session() as sess:
      augmented_tensor_dict_out = sess.run(augmented_tensor_dict)

    self.assertAllEqual(
        augmented_tensor_dict_out[fields.InputDataFields.image].shape,
        [20, 20, 3]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_boxes],
        [[10, 10, 20, 20]]
    )
    self.assertAllClose(
        augmented_tensor_dict_out[fields.InputDataFields.groundtruth_keypoints],
        [[[10, 20], [10, 10]]]
    )


def _fake_model_preprocessor_fn(image):
  return (image, tf.expand_dims(tf.shape(image)[1:], axis=0))


def _fake_image_resizer_fn(image, mask):
  return (image, mask, tf.shape(image))


def _fake_resize50_preprocess_fn(image):
  image = image[0]
  image, shape = preprocessor.resize_to_range(
      image, min_dimension=50, max_dimension=50, pad_to_max_dimension=True)

  return tf.expand_dims(image, 0), tf.expand_dims(shape, axis=0)


class DataTransformationFnTest(test_case.TestCase, parameterized.TestCase):

  def test_combine_additional_channels_if_present(self):
    image = np.random.rand(4, 4, 3).astype(np.float32)
    additional_channels = np.random.rand(4, 4, 2).astype(np.float32)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(image),
        fields.InputDataFields.image_additional_channels:
            tf.constant(additional_channels),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 1], np.int32))
    }

    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=1)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllEqual(transformed_inputs[fields.InputDataFields.image].dtype,
                        tf.float32)
    self.assertAllEqual(transformed_inputs[fields.InputDataFields.image].shape,
                        [4, 4, 5])
    self.assertAllClose(transformed_inputs[fields.InputDataFields.image],
                        np.concatenate((image, additional_channels), axis=2))

  def test_use_multiclass_scores_when_present(self):
    image = np.random.rand(4, 4, 3).astype(np.float32)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(image),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.multiclass_scores:
            tf.constant(np.array([0.2, 0.3, 0.5, 0.1, 0.6, 0.3], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 2], np.int32))
    }

    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=3, use_multiclass_scores=True)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllClose(
        np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]], np.float32),
        transformed_inputs[fields.InputDataFields.groundtruth_classes])

  def test_use_multiclass_scores_when_not_present(self):
    image = np.random.rand(4, 4, 3).astype(np.float32)
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(image),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.multiclass_scores:
            tf.placeholder(tf.float32),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 2], np.int32))
    }

    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=3, use_multiclass_scores=True)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict),
          feed_dict={
              tensor_dict[fields.InputDataFields.multiclass_scores]:
                  np.array([], dtype=np.float32)
          })

    self.assertAllClose(
        np.array([[0, 1, 0], [0, 0, 1]], np.float32),
        transformed_inputs[fields.InputDataFields.groundtruth_classes])

  @parameterized.parameters(
      {'labeled_classes': [1, 2]},
      {'labeled_classes': []},
      {'labeled_classes': [1, -1, 2]}  # -1 denotes an unrecognized class
  )
  def test_use_labeled_classes(self, labeled_classes):

    def compute_fn(image, groundtruth_boxes, groundtruth_classes,
                   groundtruth_labeled_classes):
      tensor_dict = {
          fields.InputDataFields.image:
              image,
          fields.InputDataFields.groundtruth_boxes:
              groundtruth_boxes,
          fields.InputDataFields.groundtruth_classes:
              groundtruth_classes,
          fields.InputDataFields.groundtruth_labeled_classes:
              groundtruth_labeled_classes
      }

      input_transformation_fn = functools.partial(
          inputs.transform_input_data,
          model_preprocess_fn=_fake_model_preprocessor_fn,
          image_resizer_fn=_fake_image_resizer_fn,
          num_classes=3)
      return input_transformation_fn(tensor_dict=tensor_dict)

    image = np.random.rand(4, 4, 3).astype(np.float32)
    groundtruth_boxes = np.array([[.5, .5, 1, 1], [.5, .5, 1, 1]], np.float32)
    groundtruth_classes = np.array([1, 2], np.int32)
    groundtruth_labeled_classes = np.array(labeled_classes, np.int32)

    transformed_inputs = self.execute_cpu(compute_fn, [
        image, groundtruth_boxes, groundtruth_classes,
        groundtruth_labeled_classes
    ])

    if labeled_classes == [1, 2] or labeled_classes == [1, -1, 2]:
      transformed_labeled_classes = [1, 1, 0]
    elif not labeled_classes:
      transformed_labeled_classes = [1, 1, 1]
    else:
      logging.exception('Unexpected labeled_classes %r', labeled_classes)

    self.assertAllEqual(
        np.array(transformed_labeled_classes, np.float32),
        transformed_inputs[fields.InputDataFields.groundtruth_labeled_classes])

  def test_returns_correct_class_label_encodings(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[0, 0, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }
    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_classes],
        [[0, 0, 1], [1, 0, 0]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_confidences],
        [[0, 0, 1], [1, 0, 0]])

  def test_returns_correct_labels_with_unrecognized_class(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(
                np.array([[0, 0, 1, 1], [.2, .2, 4, 4], [.5, .5, 1, 1]],
                         np.float32)),
        fields.InputDataFields.groundtruth_area:
            tf.constant(np.array([.5, .4, .3])),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, -1, 1], np.int32)),
        fields.InputDataFields.groundtruth_keypoints:
            tf.constant(
                np.array([[[.1, .1]], [[.2, .2]], [[.5, .5]]],
                         np.float32)),
        fields.InputDataFields.groundtruth_keypoint_visibilities:
            tf.constant([[True, True], [False, False], [True, True]]),
        fields.InputDataFields.groundtruth_instance_masks:
            tf.constant(np.random.rand(3, 4, 4).astype(np.float32)),
        fields.InputDataFields.groundtruth_is_crowd:
            tf.constant([False, True, False]),
        fields.InputDataFields.groundtruth_difficult:
            tf.constant(np.array([0, 0, 1], np.int32))
    }

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_classes],
        [[0, 0, 1], [1, 0, 0]])
    self.assertAllEqual(
        transformed_inputs[fields.InputDataFields.num_groundtruth_boxes], 2)
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_area], [.5, .3])
    self.assertAllEqual(
        transformed_inputs[fields.InputDataFields.groundtruth_confidences],
        [[0, 0, 1], [1, 0, 0]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_boxes],
        [[0, 0, 1, 1], [.5, .5, 1, 1]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_keypoints],
        [[[.1, .1]], [[.5, .5]]])
    self.assertAllEqual(
        transformed_inputs[
            fields.InputDataFields.groundtruth_keypoint_visibilities],
        [[True, True], [True, True]])
    self.assertAllEqual(
        transformed_inputs[
            fields.InputDataFields.groundtruth_instance_masks].shape, [2, 4, 4])
    self.assertAllEqual(
        transformed_inputs[fields.InputDataFields.groundtruth_is_crowd],
        [False, False])
    self.assertAllEqual(
        transformed_inputs[fields.InputDataFields.groundtruth_difficult],
        [0, 1])

  def test_returns_correct_merged_boxes(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        merge_multiple_boxes=True)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_boxes],
        [[.5, .5, 1., 1.]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_classes],
        [[1, 0, 1]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_confidences],
        [[1, 0, 1]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.num_groundtruth_boxes],
        1)

  def test_returns_correct_groundtruth_confidences_when_input_present(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[0, 0, 1, 1], [.5, .5, 1, 1]], np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32)),
        fields.InputDataFields.groundtruth_confidences:
            tf.constant(np.array([1.0, -1.0], np.float32))
    }
    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_classes],
        [[0, 0, 1], [1, 0, 0]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_confidences],
        [[0, 0, 1], [-1, 0, 0]])

  def test_returns_resized_masks(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(4, 4, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_instance_masks:
            tf.constant(np.random.rand(2, 4, 4).astype(np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32)),
        fields.InputDataFields.original_image_spatial_shape:
            tf.constant(np.array([4, 4], np.int32))
    }

    def fake_image_resizer_fn(image, masks=None):
      resized_image = tf.image.resize_images(image, [8, 8])
      results = [resized_image]
      if masks is not None:
        resized_masks = tf.transpose(
            tf.image.resize_images(tf.transpose(masks, [1, 2, 0]), [8, 8]),
            [2, 0, 1])
        results.append(resized_masks)
      results.append(tf.shape(resized_image))
      return results

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=fake_image_resizer_fn,
        num_classes=num_classes,
        retain_original_image=True)
    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.original_image].dtype, tf.uint8)
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.original_image_spatial_shape], [4, 4])
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.original_image].shape, [8, 8, 3])
    self.assertAllEqual(transformed_inputs[
        fields.InputDataFields.groundtruth_instance_masks].shape, [2, 8, 8])

  def test_applies_model_preprocess_fn_to_image_tensor(self):
    np_image = np.random.randint(256, size=(4, 4, 3))
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np_image),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }

    def fake_model_preprocessor_fn(image):
      return (image / 255., tf.expand_dims(tf.shape(image)[1:], axis=0))

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(transformed_inputs[fields.InputDataFields.image],
                        np_image / 255.)
    self.assertAllClose(transformed_inputs[fields.InputDataFields.
                                           true_image_shape],
                        [4, 4, 3])

  def test_applies_data_augmentation_fn_to_tensor_dict(self):
    np_image = np.random.randint(256, size=(4, 4, 3))
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np_image),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }

    def add_one_data_augmentation_fn(tensor_dict):
      return {key: value + 1 for key, value in tensor_dict.items()}

    num_classes = 4
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=add_one_data_augmentation_fn)
    with self.test_session() as sess:
      augmented_tensor_dict = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllEqual(augmented_tensor_dict[fields.InputDataFields.image],
                        np_image + 1)
    self.assertAllEqual(
        augmented_tensor_dict[fields.InputDataFields.groundtruth_classes],
        [[0, 0, 0, 1], [0, 1, 0, 0]])

  def test_applies_data_augmentation_fn_before_model_preprocess_fn(self):
    np_image = np.random.randint(256, size=(4, 4, 3))
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np_image),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([3, 1], np.int32))
    }

    def mul_two_model_preprocessor_fn(image):
      return (image * 2, tf.expand_dims(tf.shape(image)[1:], axis=0))

    def add_five_to_image_data_augmentation_fn(tensor_dict):
      tensor_dict[fields.InputDataFields.image] += 5
      return tensor_dict

    num_classes = 4
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=mul_two_model_preprocessor_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=add_five_to_image_data_augmentation_fn)
    with self.test_session() as sess:
      augmented_tensor_dict = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))

    self.assertAllEqual(augmented_tensor_dict[fields.InputDataFields.image],
                        (np_image + 5) * 2)

  def test_resize_with_padding(self):

    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(100, 50, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.0, .0, .5, .5]],
                                 np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 2], np.int32)),
        fields.InputDataFields.groundtruth_keypoints:
            tf.constant([[[0.1, 0.2]], [[0.3, 0.4]]]),
    }

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_resize50_preprocess_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_boxes],
        [[.5, .25, 1., .5], [.0, .0, .5, .25]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_keypoints],
        [[[.1, .1]], [[.3, .2]]])

  def test_groundtruth_keypoint_weights(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(100, 50, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.0, .0, .5, .5]],
                                 np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 2], np.int32)),
        fields.InputDataFields.groundtruth_keypoints:
            tf.constant([[[0.1, 0.2], [0.3, 0.4]],
                         [[0.5, 0.6], [0.7, 0.8]]]),
        fields.InputDataFields.groundtruth_keypoint_visibilities:
            tf.constant([[True, False], [True, True]]),
    }

    num_classes = 3
    keypoint_type_weight = [1.0, 2.0]
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_resize50_preprocess_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes,
        keypoint_type_weight=keypoint_type_weight)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_keypoints],
        [[[0.1, 0.1], [0.3, 0.2]],
         [[0.5, 0.3], [0.7, 0.4]]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_keypoint_weights],
        [[1.0, 0.0], [1.0, 2.0]])

  def test_groundtruth_keypoint_weights_default(self):
    tensor_dict = {
        fields.InputDataFields.image:
            tf.constant(np.random.rand(100, 50, 3).astype(np.float32)),
        fields.InputDataFields.groundtruth_boxes:
            tf.constant(np.array([[.5, .5, 1, 1], [.0, .0, .5, .5]],
                                 np.float32)),
        fields.InputDataFields.groundtruth_classes:
            tf.constant(np.array([1, 2], np.int32)),
        fields.InputDataFields.groundtruth_keypoints:
            tf.constant([[[0.1, 0.2], [0.3, 0.4]],
                         [[0.5, 0.6], [0.7, 0.8]]]),
    }

    num_classes = 3
    input_transformation_fn = functools.partial(
        inputs.transform_input_data,
        model_preprocess_fn=_fake_resize50_preprocess_fn,
        image_resizer_fn=_fake_image_resizer_fn,
        num_classes=num_classes)

    with self.test_session() as sess:
      transformed_inputs = sess.run(
          input_transformation_fn(tensor_dict=tensor_dict))
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_keypoints],
        [[[0.1, 0.1], [0.3, 0.2]],
         [[0.5, 0.3], [0.7, 0.4]]])
    self.assertAllClose(
        transformed_inputs[fields.InputDataFields.groundtruth_keypoint_weights],
        [[1.0, 1.0], [1.0, 1.0]])


class PadInputDataToStaticShapesFnTest(test_case.TestCase):

  def test_pad_images_boxes_and_classes(self):
    input_tensor_dict = {
        fields.InputDataFields.image:
            tf.placeholder(tf.float32, [None, None, 3]),
        fields.InputDataFields.groundtruth_boxes:
            tf.placeholder(tf.float32, [None, 4]),
        fields.InputDataFields.groundtruth_classes:
            tf.placeholder(tf.int32, [None, 3]),
        fields.InputDataFields.true_image_shape:
            tf.placeholder(tf.int32, [3]),
        fields.InputDataFields.original_image_spatial_shape:
            tf.placeholder(tf.int32, [2])
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6])

    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image].shape.as_list(),
        [5, 6, 3])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.true_image_shape]
        .shape.as_list(), [3])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.original_image_spatial_shape]
        .shape.as_list(), [2])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.groundtruth_boxes]
        .shape.as_list(), [3, 4])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.groundtruth_classes]
        .shape.as_list(), [3, 3])

  def test_clip_boxes_and_classes(self):
    input_tensor_dict = {
        fields.InputDataFields.groundtruth_boxes:
            tf.placeholder(tf.float32, [None, 4]),
        fields.InputDataFields.groundtruth_classes:
            tf.placeholder(tf.int32, [None, 3]),
        fields.InputDataFields.num_groundtruth_boxes:
            tf.placeholder(tf.int32, [])
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6])

    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.groundtruth_boxes]
        .shape.as_list(), [3, 4])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.groundtruth_classes]
        .shape.as_list(), [3, 3])

    with self.test_session() as sess:
      out_tensor_dict = sess.run(
          padded_tensor_dict,
          feed_dict={
              input_tensor_dict[fields.InputDataFields.groundtruth_boxes]:
                  np.random.rand(5, 4),
              input_tensor_dict[fields.InputDataFields.groundtruth_classes]:
                  np.random.rand(2, 3),
              input_tensor_dict[fields.InputDataFields.num_groundtruth_boxes]:
                  5,
          })

    self.assertAllEqual(
        out_tensor_dict[fields.InputDataFields.groundtruth_boxes].shape, [3, 4])
    self.assertAllEqual(
        out_tensor_dict[fields.InputDataFields.groundtruth_classes].shape,
        [3, 3])
    self.assertEqual(
        out_tensor_dict[fields.InputDataFields.num_groundtruth_boxes],
        3)

  def test_do_not_pad_dynamic_images(self):
    input_tensor_dict = {
        fields.InputDataFields.image:
            tf.placeholder(tf.float32, [None, None, 3]),
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[None, None])

    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image].shape.as_list(),
        [None, None, 3])

  def test_images_and_additional_channels(self):
    input_tensor_dict = {
        fields.InputDataFields.image:
            tf.placeholder(tf.float32, [None, None, 5]),
        fields.InputDataFields.image_additional_channels:
            tf.placeholder(tf.float32, [None, None, 2]),
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6])

    # pad_input_data_to_static_shape assumes that image is already concatenated
    # with additional channels.
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image].shape.as_list(),
        [5, 6, 5])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image_additional_channels]
        .shape.as_list(), [5, 6, 2])

  def test_images_and_additional_channels_errors(self):
    input_tensor_dict = {
        fields.InputDataFields.image:
            tf.placeholder(tf.float32, [None, None, 3]),
        fields.InputDataFields.image_additional_channels:
            tf.placeholder(tf.float32, [None, None, 2]),
        fields.InputDataFields.original_image:
            tf.placeholder(tf.float32, [None, None, 3]),
    }
    with self.assertRaises(ValueError):
      _ = inputs.pad_input_data_to_static_shapes(
          tensor_dict=input_tensor_dict,
          max_num_boxes=3,
          num_classes=3,
          spatial_image_shape=[5, 6])

  def test_gray_images(self):
    input_tensor_dict = {
        fields.InputDataFields.image:
            tf.placeholder(tf.float32, [None, None, 1]),
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6])

    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image].shape.as_list(),
        [5, 6, 1])

  def test_gray_images_and_additional_channels(self):
    input_tensor_dict = {
        fields.InputDataFields.image:
            tf.placeholder(tf.float32, [None, None, 3]),
        fields.InputDataFields.image_additional_channels:
            tf.placeholder(tf.float32, [None, None, 2]),
    }
    # pad_input_data_to_static_shape assumes that image is already concatenated
    # with additional channels.
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6])

    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image].shape.as_list(),
        [5, 6, 3])
    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.image_additional_channels]
        .shape.as_list(), [5, 6, 2])

  def test_keypoints(self):
    input_tensor_dict = {
        fields.InputDataFields.groundtruth_keypoints:
            tf.placeholder(tf.float32, [None, 16, 4]),
        fields.InputDataFields.groundtruth_keypoint_visibilities:
            tf.placeholder(tf.bool, [None, 16]),
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6])

    self.assertAllEqual(
        padded_tensor_dict[fields.InputDataFields.groundtruth_keypoints]
        .shape.as_list(), [3, 16, 4])
    self.assertAllEqual(
        padded_tensor_dict[
            fields.InputDataFields.groundtruth_keypoint_visibilities]
        .shape.as_list(), [3, 16])

  def test_context_features(self):
    context_memory_size = 8
    context_feature_length = 10
    max_num_context_features = 20
    input_tensor_dict = {
        fields.InputDataFields.context_features:
            tf.placeholder(tf.float32,
                           [context_memory_size, context_feature_length]),
        fields.InputDataFields.context_feature_length:
            tf.placeholder(tf.float32, [])
    }
    padded_tensor_dict = inputs.pad_input_data_to_static_shapes(
        tensor_dict=input_tensor_dict,
        max_num_boxes=3,
        num_classes=3,
        spatial_image_shape=[5, 6],
        max_num_context_features=max_num_context_features,
        context_feature_length=context_feature_length)

    self.assertAllEqual(
        padded_tensor_dict[
            fields.InputDataFields.context_features].shape.as_list(),
        [max_num_context_features, context_feature_length])

    with self.test_session() as sess:
      feed_dict = {
          input_tensor_dict[fields.InputDataFields.context_features]:
              np.ones([context_memory_size, context_feature_length],
                      dtype=np.float32),
          input_tensor_dict[fields.InputDataFields.context_feature_length]:
              context_feature_length
      }
      padded_tensor_dict_out = sess.run(padded_tensor_dict, feed_dict=feed_dict)

    self.assertEqual(
        padded_tensor_dict_out[fields.InputDataFields.valid_context_size],
        context_memory_size)


if __name__ == '__main__':
  tf.test.main()
