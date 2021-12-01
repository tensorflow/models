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

from official.vision.beta.projects.mesh_rcnn.data import create_pix3d_tf_record

def get_features(tfrecord_filename, num_models):
  raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
  for raw_record in raw_dataset.take(num_models):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature
  
  return features

class TFRecordGeneratorTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
    ((3,3), [[0.7829390757301289, 0.005066182290536127, 0.6220777583966899], 
    [-0.31879059850064173, 0.8619595114951648, 0.39420598023193354], 
    [-0.5342087213837434, -0.506951806723298, 0.676476834531331]])
  )
  def test_rot_mat(self, expected_shape: tuple,
                   expected_output: list):
    
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
    
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename, 
                            num_models=1)

    trans_mat = features['camera/trans_mat'].float_list.value
    
    self.assertAllEqual(expected_shape, np.shape(trans_mat))
    self.assertAllEqual((np.array(expected_output)).astype(np.float32), trans_mat)


  @parameterized.parameters(
  ((4,), [320, 56, 2131, 3609])
  )
  def test_bbox(self, expected_shape: tuple,
                expected_output: list):
    
    tfrecord_filename = 'tmp-00000-of-00001.tfrecord'
    features = get_features(tfrecord_filename=tfrecord_filename, 
                            num_models=1)

    bbox = features['image/object/bbox'].int64_list.value
    
    self.assertAllEqual(expected_shape, np.shape(bbox))
    self.assertAllEqual((np.array(expected_output)).astype(np.int64), bbox)

if __name__ == '__main__':
  # make output file directory if it doesn't exist
  directory = os.path.dirname("tmp")
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  pix3d_dir = 'D:\Datasets\pix3d'
  #create TF record files 
  create_pix3d_tf_record._create_tf_record_from_pix3d_dir(
      pix3d_dir, "tmp", 1,
      "tfrecord_pix3d_test_annotations.json")

  tf.test.main()

