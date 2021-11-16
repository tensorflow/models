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

r"""Takes TFRecord containing Pix3D dataset and visualizes it.
Example usage:
    python pix3d_visualizer.py --logtostderr \
      --num_models=2 \
      --pix3d_records_dir="${PIX3D_RECORDS_DIR}" \
      --output_folder="${OUTPUT_DIR}"
"""

import logging
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from absl import app, flags

flags.DEFINE_multi_string(
    'pix3d_records_dir', '', 'Directory containing Pix3d TFRecords.')
flags.DEFINE_string(
    'output_folder', '/tmp/output', 'Path to output files')
flags.DEFINE_integer(
    'num_models', 2, 'Number of models rebuilt from TFRecord.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

def write_obj_file(vertices: List[float], faces: List[int], filename: str):
  """Writes a new .obj file using the vertices and faces of a mesh.

  Args:
    vertices: List of vertices, where each element contains 3 floats for the
        x, y, z coordinates of the vertex.
    faces: List of faces, where each element contains 3 integers that represent
        the vertices corresponding to this face.
    filename: String for the filename of the .obj file.
  """
  logging.info('Writing mesh to: %s', filename)
  with open(filename, 'w') as f:
    for vertex in vertices:
      f.write("v " + ' '.join(map(str, vertex)))

    for face in faces:
      f.write("f " + ' '.join(map(str, face)))

def write_masked_image(mask: List[List[int]], image: List[List[List[int]]],
                       filename: str):
  """Writes a new .png file using an image overlayed with a mask.

  Args:
      mask: A 2D list containing the mask data
      image: A 3D list containing the image data
      filename: String for the filename of the .png file.
  """
  logging.info('Writing mask image to: %s', filename)
  dim1 = len(mask)
  dim2 = len(mask[0])

  for i in range(dim1):
    for j in range(dim2):
      # add green mask to image
      if mask[i][j] > 0:
        image[i][j][1] += 50
        image[i][j][1] = max(image[i][j][1], 255)

  cv2.imwrite(filename, np.array(image))

def write_voxel_files(encoded_voxel_data: bytes, mat_filename: str,
                      render_filename: str):
  """Writes a new .mat file using the voxel data and renders it as a .png file.

  Args:
      encoded_voxel_data: Bytes object with encoded voxel data.
      mat_filename: String for the filename of the .mat file.
      render_filename: String for the filename of the .png file.
  """
  logging.info('Writing voxel data to: %s', mat_filename)
  with open(mat_filename, 'wb') as f:
    f.write(encoded_voxel_data)

  # Loading voxel data as a 3D array
  mat_contents = sio.loadmat(mat_filename)
  voxel = mat_contents['voxel']

  logging.info('Writing voxel render to: %s', render_filename)

  # Visualize and render the voxels
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.voxels(voxel)
  plt.savefig(render_filename)

def visualize_tf_record(pix3d_records_dir: str,
                        output_path: str,
                        num_models: int):
  """Visualizes Pix3D data from TFRecords.

  Args:
    pix3d_records_dir: Path to a directory containing the Pix3D TFRecords.
    output_path: Path to output the visualization files.
    num_models: Number of output files to create.
  """

  logging.info(
      'Begin visualize %d models from the TFRecords in directory %s \
       into %s.', num_models, pix3d_records_dir, output_path)

  filenames = [
      os.path.join(pix3d_records_dir, x) for x in os.listdir(pix3d_records_dir)]

  raw_dataset = tf.data.TFRecordDataset(filenames)

  # Iterate through some samples in the dataset
  for raw_record in raw_dataset.take(num_models):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature

    # Extract the information to visualize
    vertices = tf.io.parse_tensor(
        features['model/vertices'].bytes_list.value[0],
        tf.float32).numpy().tolist()
    faces = tf.io.parse_tensor(
        features['model/faces'].bytes_list.value[0], tf.int32).numpy().tolist()
    mask = cv2.imdecode(
        np.fromstring(features['image/object/mask'].bytes_list.value[0],
                      np.uint8),
        cv2.IMREAD_GRAYSCALE).tolist()
    image = cv2.imdecode(
        np.fromstring(features['image/encoded'].bytes_list.value[0], np.uint8),
        flags=1).tolist()

    filename = str(
        features['image/filename'].bytes_list.value[0]).split("/")[2][:-1]
    filename = filename.split(".")[0]
    filename = os.path.join(output_path, filename)

    encoded_voxel_data = features['model/voxel'].bytes_list.value[0]

    # Write the features
    write_obj_file(vertices, faces, filename + '.obj')
    write_masked_image(mask, image, filename + '.png')
    write_voxel_files(encoded_voxel_data, filename + '.mat',
                      filename + '_rendered' + '.png')

def main(_):
  assert FLAGS.pix3d_records_dir, '`pix3d_dir` missing.'

  directory = os.path.dirname(FLAGS.output_folder)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  visualize_tf_record(
      FLAGS.pix3d_records_dir[0], FLAGS.output_folder, FLAGS.num_models)

if __name__ == '__main__':
  app.run(main)
