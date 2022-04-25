# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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


def get_features_single_instance(tfrecord_filename: str):
  """Grabs the features from a single sample from a TFRecord.

  Args:
    tfrecord_filename: String for the filename of the TFRecord.

  Returns:
    features: a tf.train.Feature message that holds the feature data.
  """
  raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
  for raw_record in raw_dataset.take(1):
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
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

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
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

    trans_mat = features['camera/trans_mat'].float_list.value

    self.assertAllEqual(expected_shape, np.shape(trans_mat))
    self.assertAllEqual(
        (np.array(expected_output)).astype(np.float32), trans_mat)

  @parameterized.parameters(
      ((320, 56, 2131, 3609),)
  )
  def test_bbox(self, expected_output: list):
    """Test for bounding boxes.

    Args:
      expected_output: The expected values of the bounding box.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

    xmin = features['image/object/bbox/xmin'].float_list.value[0]
    ymin = features['image/object/bbox/ymin'].float_list.value[0]
    xmax = features['image/object/bbox/xmax'].float_list.value[0]
    ymax = features['image/object/bbox/ymax'].float_list.value[0]

    self.assertAllEqual(
        (np.array(expected_output)).astype(np.int64), [xmin, ymin, xmax, ymax])

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
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

    intrinstic_mat = features['camera/intrinstic_mat'].float_list.value

    self.assertAllEqual(expected_shape, np.shape(intrinstic_mat))
    self.assertAllEqual(
        (np.array(expected_output)).astype(np.float32), intrinstic_mat)

  def test_voxel(self):
    """Test for encoded voxels."""
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

    voxel_coords = tf.io.parse_tensor(
        features['model/voxel_indices'].bytes_list.value[0],
        tf.int64).numpy()
    voxel_shape = features['model/voxel_shape'].int64_list.value

    self.assertAllEqual(voxel_shape, [128, 128, 128])
    self.assertAllEqual(np.shape(voxel_coords)[1:], [3])

  @parameterized.parameters(
      ((49840, 3),)
  )
  def test_vertices(self, expected_shape: tuple):
    """Test for vertices.

    Args:
      expected_shape: The expected shape of vertices.
    """
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

    vertices = tf.io.parse_tensor(
        features['model/vertices'].bytes_list.value[0],
        tf.float64).numpy()

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
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)

    faces = tf.io.parse_tensor(
        features['model/faces'].bytes_list.value[0],
        tf.int32).numpy()

    self.assertAllEqual(expected_shape, np.shape(faces))

  def test_image(self):
    """Test for image."""
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)
    image = cv2.imdecode(
        np.fromstring(features['image/encoded'].bytes_list.value[0], np.uint8),
        flags=1)

    self.assertGreater(np.shape(image)[0], 0)
    self.assertGreater(np.shape(image)[1], 0)
    self.assertEqual(np.shape(image)[2], 3)

  def test_mask(self):
    """Test for mask."""
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)
    mask = tf.io.decode_png(
        features['image/encoded'].bytes_list.value[0],
        channels=1, dtype=tf.uint8)

    self.assertGreater(np.shape(mask)[0], 0)
    self.assertGreater(np.shape(mask)[1], 0)
    self.assertEqual(np.shape(mask)[2], 1)

  def test_class(self):
    """Test for class."""
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features_single_instance(tfrecord_filename=tfrecord_filename)
    class_id = features['image/object/class/label'].int64_list.value[0]

    self.assertEqual(class_id, 3)

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
  pix3d_dir = r'C:\ML\Datasets\pix3d'

  # Create TF record files
  _create_tf_record_from_pix3d_dir(
      pix3d_dir, "tmp", 1,
      "tfrecord_pix3d_test_annotations.json")

  tf.test.main()
