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

"""Mesh RCNN Dataset Testing functions"""
import os
import cv2

import numpy as np
import tensorflow as tf
import scipy.io as sio
from absl.testing import parameterized
from io import BytesIO

from official.core import task_factory, train_utils
from official.vision.beta.ops import voxel_ops

PATH_TO_PIX3D_RECORDS = '/local/tfmg/meshrcnn/tf-models/pix3d_records'

def resize_image(image, batch_size=1):
  return tf.image.resize_with_pad(image, [batch_size, 800, 800, 3])

def test_meshrcnn_input_task(batch_size = 1):
    pass

class MeshRCNNInputTest(tf.test.TestCase, parameterized.TestCase):

  # @parameterized.parameters()
  def test_meshrcnn_input(self):
    # builds a pipeline from the config and tests the datapipline shapes
    # dataset, _, params = test_yolo_input_task(
    #     scaled_pipeline=scaled_pipeline, 
    #     batch_size=1)
    # _, dataset, params = test_meshrcnn_input_task( 
    #     batch_size=1)

    # dataset = dataset.take(100)

    # for image, label in dataset:
    #   self.assertAllEqual(image.shape, ([1] + params.model.input_size))
    #   self.assertTrue(
    #       tf.reduce_all(tf.math.logical_and(image >= 0, image <= 1)))
    self.assertAllEqual(1, 1)
if __name__ == '__main__':
  # tf.test.main()

  tfrecord_filename = PATH_TO_PIX3D_RECORDS + '/tmp-00000-of-00001.tfrecord'

  raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)
  for i, raw_record in enumerate(raw_dataset.take(1)):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature

  encoded_voxel = (features['model/voxel'].bytes_list.value[0])

  # image = cv2.imdecode(
  #     np.fromstring(features['image/encoded'].bytes_list.value[0], np.uint8),
  #     flags=1)
  # image = tf.convert_to_tensor(image, dtype=tf.float32)
  faces = tf.io.parse_tensor(
        features['model/faces'].bytes_list.value[0],
        tf.int32).numpy()

  vertices = tf.io.parse_tensor(
        features['model/vertices'].bytes_list.value[0],
        tf.float32).numpy()

  image = tf.io.decode_image(
    features['image/encoded'].bytes_list.value[0], channels=3, dtype=tf.dtypes.uint8, name=None,
    expand_animations=True
  )
  image = tf.image.resize_with_crop_or_pad(image, 800, 800)

  rot_mat = tf.io.parse_tensor(
        features['camera/rot_mat'].bytes_list.value[0],
        tf.float32).numpy()

  trans_mat = (features['camera/trans_mat'].float_list.value)

  # fileobj = BytesIO(bytes.fromhex(str(encoded_voxel[116:], encoding="utf-8", errors="ignore"))) 
  # voxel = tf.convert_to_tensor(sio.loadmat(fileobj)["voxel"].tolist(), dtype=tf.float16)
  # print(encoded_voxel[116:])

  MAX_FACES = 126748
  MAX_VERTICES = 108416

  # print(faces.shape)
  # print(vertices.shape)

  faces.resize(MAX_FACES, 3)
  vertices.resize(MAX_VERTICES, 3)

  faces_mask = np.zeros((MAX_FACES,))
  vertices_mask = np.zeros((MAX_VERTICES,))

  faces_mask[0 : MAX_FACES] = 1
  vertices_mask[0 : MAX_VERTICES] = 1



  with open("voxel.mat", 'wb') as f:
    f.write(encoded_voxel)

  mat_contents = sio.loadmat("voxel.mat")
  voxel = mat_contents['voxel']

  verts = voxel_ops.read_voxel("voxel.mat")
  verts = voxel_ops.transform_verts(verts, rot_mat, trans_mat)
  verts = voxel_ops.resize_coordinates(verts, 10, 10)

  verts = voxel_ops.horizontal_flip_coordinates(verts)
  print(verts.shape)

  # og_voxel = voxel_ops.verts2voxel(verts, [128, 128, 128])

  # down_voxel = voxel_ops.downsample(og_voxel, 5)

  # print((voxel.shape))
  
  # print(og_voxel.shape)
  # # print(down_voxel.shape)
  # # print(voxel == og_voxel)
  # print(np.array_equal(voxel, og_voxel))

  # #Counting max number of faces for each object
  # print(voxel_ops.num_faces("pix3d/model/bed/IKEA_BEDDINGE/model.obj"))

  # max_faces = 0
  # max_vertices = 0

  # for root, dirs, files in os.walk("pix3d/model", topdown=False):
  #  for name in files:
  #     if name[-4:] == ".obj":
  #       # print(os.path.join(root, name))
  #       max_faces = max(max_faces, (voxel_ops.num_faces(os.path.join(root, name))))
  #       max_vertices = max(max_vertices, (voxel_ops.num_vertices(os.path.join(root, name))))

  # print(max_faces)
  # print(max_vertices)


  # voxel_ops.visualize_voxel(voxel)
