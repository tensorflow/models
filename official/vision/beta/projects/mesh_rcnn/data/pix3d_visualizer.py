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
      --pix3d_dir="${PIX3D_TFRECORD_DIR}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}"    
"""

import logging
import os

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np
import cv2

import tensorflow as tf

flags.DEFINE_multi_string('pix3d_dir', '', 'Directory containing Pix3d.')
flags.DEFINE_string('output_file_prefix', '/tmp/output', 'Path to output files')
flags.DEFINE_integer('num_models', 2, 'Number of models rebuilt from TFRecord.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def write_obj_file(vertices,
                   faces,
                   filename):
    """Writes a new .obj file from data.
    Args:
        verticies: List of vertices
        faces: List of faces
        filename: Filename to write .obj file to
    """
    logging.info(f"Logging file {filename}")
    with open(filename, 'w') as f:
        for vertex in vertices:
            print("v " + ' '.join(map(str, vertex)), file=f)

        for face in faces:
            ret = "f "
            
            for coordinate in face:
                ret += str(coordinate[0]) + " "

            print(ret, file=f)

def write_masked_image(mask,
                       image,
                       filename):
    """Writes a new .png file from data.
    Args:
        mask: 2d array of mask data
        image: 3d array of image data
        filename: Filename to write .png file to
    """
    logging.info(f"Logging file {filename}")
    dim1 = len(mask)
    dim2 = len(mask[0])
    
    for i in range (dim1):
        for j in range (dim2):
            if mask[i][j] > 0: # add green mask to image
                image[i][j][1] += 50

    cv2.imwrite(filename, np.array(image))
    

def visualize_tf_record(pix3d_dir,
                        output_path,
                        num_models):
    """Visualizes pix3d data in TFRecord format.
    Args:
      pix3d_dir: pix3d_dir for TFRecords
      output_path: Path to output .obj files.
      num_models: Number of output files to create.
    """

    logging.info(f"Starting to visualize {num_models} models from the TFRecords in directory {pix3d_dir} into {output_path}.")

    filenames = [os.path.join(pix3d_dir, x) for x in os.listdir(pix3d_dir)]
    
    raw_dataset = tf.data.TFRecordDataset(filenames)
    
    for raw_record in raw_dataset.take(num_models):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        vertices = tf.io.parse_tensor(features["model/vertices"].bytes_list.value[0], tf.float32).numpy().tolist()
        faces = tf.io.parse_tensor(features["model/faces"].bytes_list.value[0], tf.int32).numpy().tolist()
        mask = cv2.imdecode(np.fromstring(features["mask"].bytes_list.value[0], np.uint8), cv2.IMREAD_GRAYSCALE).tolist()
        image = cv2.imdecode(np.fromstring(features["img/encoded"].bytes_list.value[0], np.uint8), flags=1).tolist()
        
        #mask = tf.io.parse_tensor(features["mask"].bytes_list.value[0], tf.int32).numpy().tolist()
        #image = tf.io.parse_tensor(features["img/encoded"].bytes_list.value[0], tf.int32).numpy().tolist()

        filename = str(features["img/filename"].bytes_list.value[0]).split("/")[2][:-1]
        filename = filename.split(".")[0]
        filename = os.path.join(output_path, filename)

        write_obj_file(vertices, faces, filename + ".obj")
        write_masked_image(mask, image, filename + ".png")

def main(_):
    assert FLAGS.pix3d_dir, '`pix3d_dir` missing.'

    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)

    visualize_tf_record(FLAGS.pix3d_dir[0], FLAGS.output_file_prefix, FLAGS.num_models)

    
if __name__ == '__main__':
    app.run(main)
