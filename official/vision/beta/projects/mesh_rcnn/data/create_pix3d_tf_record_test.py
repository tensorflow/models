# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for create_pix3d_tf_record.py."""
import os

import cv2
import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.mesh_rcnn.data.create_pix3d_tf_record import \
    _create_tf_record_from_pix3d_dir


def get_features(tfrecord_filename, num_models):
  raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
  for raw_record in raw_dataset.take(num_models):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature

  return features

class TFRecordGeneratorTest(parameterized.TestCase, tf.test.TestCase):
  """Test cases for TFRecord Generator."""
  @parameterized.parameters(
      ((3, 3), [[0.7829390757301289, 0.005066182290536127, 0.6220777583966899],
                [-0.31879059850064173, 0.8619595114951648, 0.39420598023193354],
                [-0.5342087213837434, -0.506951806723298, 0.676476834531331]])
  )
  def test_rot_mat(self, expected_shape: tuple,
                   expected_output: list):
    """Test for rotation matrix.
    Args:
      expected_shape: The expected shape of the rotation matrix.
      expected_output: The expected values of the rotation matrix.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    rot_mat = tf.io.parse_tensor(
        features['camera/rot_mat'].bytes_list.value[0],
        tf.float32).numpy()

    self.assertAllEqual(expected_shape, np.shape(rot_mat))
    self.assertAllEqual((np.array(expected_output)).astype(np.float32), rot_mat)

  @parameterized.parameters(
      ((3,), [0.017055282615799992, 0.058128800728, 0.938664993887])
  )
  def test_trans_mat(self, expected_shape: tuple,
                     expected_output: list):
    """Test for translation matrix.
    Args:
      expected_shape: The expected shape of the translation matrix.
      expected_output: The expected values of the translation matrix.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    trans_mat = features['camera/trans_mat'].float_list.value

    self.assertAllEqual(expected_shape, np.shape(trans_mat))
    self.assertAllEqual(
        (np.array(expected_output)).astype(np.float32), trans_mat)


  @parameterized.parameters(
      ((4,), [320, 56, 2131, 3609])
  )
  def test_bbox(self, expected_shape: tuple,
                expected_output: list):
    """Test for bounding boxes.
    Args:
      expected_shape: The expected shape of the bounding box.
      expected_output: The expected values of the bounding box.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    bbox = features['image/object/bbox'].int64_list.value

    self.assertAllEqual(expected_shape, np.shape(bbox))
    self.assertAllEqual((np.array(expected_output)).astype(np.int64), bbox)

  @parameterized.parameters(
      ((3,), [3434.68173275, 1512.0, 2016.0])
  )
  def test_intrinsic_mat(self, expected_shape: tuple,
                         expected_output: list):
    """Test for camera intrinsic matrix.
    Args:
      expected_shape: The expected shape of the intrinsic matrix.
      expected_output: The expected values of the intrinsic matrix.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    intrinstic_mat = features['camera/intrinstic_mat'].float_list.value

    self.assertAllEqual(expected_shape, np.shape(intrinstic_mat))
    self.assertAllEqual(
        (np.array(expected_output)).astype(np.float32), intrinstic_mat)


  @parameterized.parameters(
      ((1,), [0])
  )
  def test_is_crowd(self, expected_shape: tuple,
                    expected_output: list):
    """Test for is_crowd annotation.
    Args:
      expected_shape: The expected shape of is_crowd.
      expected_output: The expected value of is_crowd.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    is_crowd = features['is_crowd'].int64_list.value

    self.assertAllEqual(expected_shape, np.shape(is_crowd))
    self.assertAllEqual(expected_output, is_crowd)

  def test_voxel(self):
    """Test for encoded voxels."""
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    voxel_len = len(features['model/voxel'].bytes_list.value[0])

    self.assertGreater(voxel_len, 0)

  @parameterized.parameters(
      ((49840, 3),)
  )
  def test_vertices(self, expected_shape: tuple):
    """Test for vertices.
    Args:
      expected_shape: The expected shape of vertices.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    vertices = tf.io.parse_tensor(
        features['model/vertices'].bytes_list.value[0],
        tf.float32).numpy()

    self.assertAllEqual(expected_shape, np.shape(vertices))

  @parameterized.parameters(
      ((100000, 3),)
  )
  def test_faces(self, expected_shape: tuple):
    """Test for faces.
    Args:
      expected_shape: The expected shape of faces.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)

    faces = tf.io.parse_tensor(
        features['model/faces'].bytes_list.value[0],
        tf.int32).numpy()

    self.assertAllEqual(expected_shape, np.shape(faces))

  def test_image(self):
    """Test for image."""
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename,
                            num_models=1)
    image = cv2.imdecode(
        np.fromstring(features['image/encoded'].bytes_list.value[0], np.uint8),
        flags=1)

    self.assertGreater(np.shape(image)[0], 0)
    self.assertGreater(np.shape(image)[1], 0)
    self.assertAllEqual(np.shape(image)[2], 3)

  @classmethod
  def tearDownClass(cls):
    """Deletes the temporary tfrecord file."""
    tf.io.gfile.remove('tmp-00000-of-00001.tfrecord')

if __name__ == '__main__':
  # Create output directory if it doesn't exist
  directory = os.path.dirname("tmp")
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  # Specify the dataset download directory
  pix3d_dir = 'pix3d'

  # Create TF record files
  _create_tf_record_from_pix3d_dir(
      pix3d_dir, "tmp", 1,
      "tfrecord_pix3d_test_annotations.json")

  tf.test.main()
