# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert raw cityscale dataset to TFRecord format.

This scripts follows the label map decoder format and supports detection
boxes, instance masks and captions.

Example usage:
    python create_cityscale_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""

import collections
import json
import logging
import os

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

import tensorflow as tf

import multiprocessing as mp
from official.vision.data import tfrecord_lib


flags.DEFINE_string('image_dir', './', 'Directory containing images.')
flags.DEFINE_string(
    'image_info_file', './data_split.json', 'File containing image information. '
    'Tf Examples in the output files correspond to the image '
    'info entries in this file. If this file is not provided '
    'object_annotations_file is used if present. Otherwise, '
    'caption_annotations_file is used to get image info.')
flags.DEFINE_string('output_file_prefix', './tfrecord/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 18, 'Number of shards for output file.')
_NUM_PROCESSES = flags.DEFINE_integer(
    'num_processes', None,
    ('Number of parallel processes to use. '
     'If set to 0, disables multi-processing.'))


FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

def _load_graph(image_dirs, filename):
  graph_path = os.path.join(image_dirs,f'graph/{filename}.json')
  with tf.io.gfile.GFile(graph_path, 'r') as fid:
    graph_data = json.load(fid)
  
  edges_id = []
  edges_src = []
  edges_dst = []
  edges_vertices = None
  edges_orientations = None
  edges_len = []
  vertices_id = []
  vertices_x = []
  vertices_y = []
  
  for e in graph_data['edges']:
    edges_id.append(e["id"])
    edges_src.append(e["src"])
    edges_dst.append(e["dst"])    
    if edges_vertices is None:
      edges_vertices = e["vertices"]
      edges_orientations = e["orientation"]
    else:
      edges_vertices = tf.concat([edges_vertices, e["vertices"]], 0)
      edges_orientations = tf.concat([edges_orientations, e["orientation"]], 0)
    edges_len.append(len(e["vertices"]))
  
  for v in graph_data['vertices']:
    vertices_id.append(v["id"])
    vertices_x.append(v["x"])
    vertices_y.append(v["y"])
  
  return (edges_id, edges_src, edges_dst, edges_vertices, edges_orientations,
          edges_len, vertices_id, vertices_x, vertices_y)

def create_tf_example(filename,
                      image_dirs):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    image_dirs: list of directories containing the image files.

  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG,
      does not exist, or is not unique across image directories.
  """
  image_path = os.path.join(image_dirs,f'20cities/region_{filename}_sat.png')
  with tf.io.gfile.GFile(image_path, 'rb') as fid:
    encoded_sat = fid.read()
  
  intsec_path = os.path.join(image_dirs,f'intersection/{filename}.png')
  with tf.io.gfile.GFile(intsec_path, 'rb') as fid:
    encoded_intsec = fid.read()
    
  segment_path = os.path.join(image_dirs,f'segment/{filename}.png')
  with tf.io.gfile.GFile(segment_path, 'rb') as fid:
    encoded_segment = fid.read()
  
  image_path = os.path.join(image_dirs,f'20cities/region_{filename}_sat.png')
  with tf.io.gfile.GFile(image_path, 'rb') as fid:
    encoded_sat = fid.read()
  
  graph_output = _load_graph(image_dirs, filename)
  
  (edges_id, edges_src, edges_dst, edges_vertices, edges_orientations,
  edges_len, vertices_id, vertices_x, vertices_y) = graph_output
  
  feature_dict = {
    "image/encoded": tfrecord_lib.convert_to_feature(encoded_sat),
    "image/intersection": tfrecord_lib.convert_to_feature(encoded_intsec),
    "image/segment": tfrecord_lib.convert_to_feature(encoded_segment),
    "edges/id": tfrecord_lib.convert_to_feature(edges_id),
    "edges/src": tfrecord_lib.convert_to_feature(edges_src),
    "edges/dst": tfrecord_lib.convert_to_feature(edges_dst),
    "edges/vertices": tfrecord_lib.convert_to_feature(
        edges_vertices, 'int64_list'),
    "edges/orientation": tfrecord_lib.convert_to_feature(
        edges_orientations, 'int64_list'),
    "edges/length": tfrecord_lib.convert_to_feature(edges_len),
    "vertices/id": tfrecord_lib.convert_to_feature(vertices_id),
    "vertices/x": tfrecord_lib.convert_to_feature(vertices_x),
    "vertices/y": tfrecord_lib.convert_to_feature(vertices_y),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  
  return example, 0

def generate_annotations(images, image_dirs):
  """Generator for cityscale annotations."""
  for image in images:
    yield (image, image_dirs)

def _create_tf_record_from_data_split(split_info_file,
                                            image_dirs,
                                            output_path,
                                            num_shards):
  """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    images_info_file: JSON file containing image info. The number of tf.Examples
      in the output tf Record files is exactly equal to the number of image info
      entries in this file. This can be any of train/val/test annotation json
      files Eg. 'image_info_test-dev2017.json',
      'instance_annotations_train2017.json',
      'caption_annotations_train2017.json', etc.
    image_dirs: List of directories containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: Number of output files to create.
  """

  logging.info('writing to output path: %s', output_path)
  # iterate training data
  with open(split_info_file,'r') as jf:
    tile_list = json.load(jf)['train']
  
  cityscale_annotations_iter = generate_annotations(tile_list, image_dirs)
  
  num_skipped = tfrecord_lib.write_tf_record_dataset(
      output_path, cityscale_annotations_iter, create_tf_example, num_shards,
      multiple_processes=_NUM_PROCESSES.value)

  logging.info('Finished writing')


def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert (FLAGS.image_info_file or FLAGS.object_annotations_file or
          FLAGS.caption_annotations_file), ('All annotation files are '
                                            'missing.')
  images_info_file = FLAGS.image_info_file

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  _create_tf_record_from_data_split(images_info_file, FLAGS.image_dir,
                                          FLAGS.output_file_prefix,
                                          FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)
