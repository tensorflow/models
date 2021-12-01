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
import numpy as np
import tensorflow as tf
import scipy.io as sio
import os

from absl.testing import parameterized
from absl import flags

# from official.vision.beta.projects.mesh_rcnn.data import create_pix3d_tf_record
# from create_pix3d_tf_record import _create_tf_record_from_pix3d_dir

def get_features(tfrecord_filename, num_models):
  raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
  for raw_record in raw_dataset.take(num_models):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature
  
  return features

class TFRecordGeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
    ((3,3), [[-0.8679031454645554, -0.012174281185507776, 0.49658424961968634], 
    [-0.23962605126130768, 0.8859471856705277, -0.3970855572283948], 
    [-0.4351131871704694, -0.4636263269919945, -0.7718336240863495]])
  )
  def test_rot_mat(self, expected_shape: tuple,
                   expected_output: list):
    
    tfrecord_filename = 'pix3d_records/tmp-00000-of-00032.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename, 
                            num_models=1)

    rot_mat = tf.io.parse_tensor(
        features['camera/rot_mat'].bytes_list.value[0],
        tf.float32).numpy()
    
    self.assertAllEqual(expected_shape, np.shape(rot_mat))
    self.assertAllEqual((np.array(expected_output)).astype(np.float32), rot_mat)

  @parameterized.parameters(
    ((3,), [0.0123832253541, -0.005298808760219998, 0.982734527031])
  )
  def test_trans_mat(self, expected_shape: tuple,
                     expected_output: list):
    
    tfrecord_filename = 'pix3d_records/tmp-00000-of-00032.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename, 
                            num_models=1)

    trans_mat = tf.io.parse_tensor(
        features['camera/trans_mat'].bytes_list.value[0],
        tf.float32).numpy()
    
    self.assertAllEqual(expected_shape, np.shape(trans_mat))
    self.assertAllEqual((np.array(expected_output)).astype(np.float32), trans_mat)


  @parameterized.parameters(
  ((4,), [231, 446, 2273, 3312])
  )
  def test_bbox(self, expected_shape: tuple,
                expected_output: list):
    
    tfrecord_filename = 'pix3d_records/tmp-00000-of-00032.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename, 
                            num_models=1)

    bbox = tf.io.parse_tensor(
        features['image/object/bbox'].bytes_list.value[0],
        tf.int32).numpy()
    
    self.assertAllEqual(expected_shape, np.shape(bbox))
    self.assertAllEqual((np.array(expected_output)).astype(np.int32), bbox)

if __name__ == '__main__':

  # #make output file directory if it doesn't exist
  # directory = os.path.dirname("tmp")
  # if not tf.io.gfile.isdir(directory):
  #   tf.io.gfile.makedirs(directory)

  # #create TF record files 
  # _create_tf_record_from_pix3d_dir(
  #     "pix3d", "tmp", 32,
  #     "pix3d_single_example.json")

  tf.test.main()

