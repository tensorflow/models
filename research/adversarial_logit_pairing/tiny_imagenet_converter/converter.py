# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Converts Tiny Imagenet dataset into TFRecord format.

As an output this program generates following files in TFRecord format:
- train.tfrecord
- validation.tfrecord
- test.tfrecord

Generated train and validation files will contain tf.Example entries with
following features:
- image/encoded - encoded image
- image/format - image format
- label/wnid - label WordNet ID
- label/imagenet - imagenet label [1 ... 1000]
- label/tiny_imagenet - tiny imagenet label [0 ... 199]
- bbox/xmin
- bbox/ymin
- bbox/xmax
- bbox/ymax

Test file will contain entries with 'image/encoded' and 'image/format' features.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
import random

from absl import app
from absl import flags
from absl import logging

import pandas as pd

import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', '', 'Input directory')
flags.DEFINE_string('output_dir', '', 'Output directory')

flags.DEFINE_string('imagenet_synsets_path', '',
                    'Optional path to /imagenet_lsvrc_2015_synsets.txt')


ImageMetadata = namedtuple('ImageMetadata', ['label', 'x1', 'y1', 'x2', 'y2'])


class WnIdToNodeIdConverter(object):
  """Converts WordNet IDs to numerical labels."""

  def __init__(self, wnids_path, background_class):
    self._wnid_to_node_id = {}
    self._node_id_to_wnid = {}
    with tf.gfile.Open(wnids_path) as f:
      wnids_sequence = [wnid.strip() for wnid in f.readlines() if wnid.strip()]
    node_id_offset = 1 if background_class else 0
    for i, label in enumerate(wnids_sequence):
      self._wnid_to_node_id[label] = i + node_id_offset
      self._node_id_to_wnid[i + node_id_offset] = label

  def to_node_id(self, wnid):
    return self._wnid_to_node_id[wnid]

  def to_wnid(self, node_id):
    return self._node_id_to_wnid[node_id]

  def all_wnids(self):
    return self._wnid_to_node_id.keys()


def read_tiny_imagenet_annotations(annotations_filename,
                                   images_dir,
                                   one_label=None):
  """Reads one file with Tiny Imagenet annotations."""
  result = []
  if one_label:
    column_names = ['filename', 'x1', 'y1', 'x2', 'y2']
  else:
    column_names = ['filename', 'label', 'x1', 'y1', 'x2', 'y2']
  with tf.gfile.Open(annotations_filename) as f:
    data = pd.read_csv(f, sep='\t', names=column_names)
  for row in data.itertuples():
    label = one_label if one_label else getattr(row, 'label')
    full_filename = os.path.join(images_dir, getattr(row, 'filename'))
    result.append((full_filename,
                   ImageMetadata(label=label,
                                 x1=getattr(row, 'x1'),
                                 y1=getattr(row, 'y1'),
                                 x2=getattr(row, 'x2'),
                                 y2=getattr(row, 'y2'))))
  return result


def read_validation_annotations(validation_dir):
  """Reads validation data annotations."""
  return read_tiny_imagenet_annotations(
      os.path.join(validation_dir, 'val_annotations.txt'),
      os.path.join(validation_dir, 'images'))


def read_training_annotations(training_dir):
  """Reads training data annotations."""
  result = []
  sub_dirs = tf.gfile.ListDirectory(training_dir)
  for sub_dir in sub_dirs:
    if not sub_dir.startswith('n'):
      logging.warning('Found non-class directory in training dir: %s', sub_dir)
      continue
    sub_dir_results = read_tiny_imagenet_annotations(
        os.path.join(training_dir, sub_dir, sub_dir + '_boxes.txt'),
        os.path.join(training_dir, sub_dir, 'images'),
        one_label=sub_dir)
    result.extend(sub_dir_results)
  return result


def read_test_annotations(test_dir):
  """Reads test data annotations."""
  files = tf.gfile.ListDirectory(os.path.join(test_dir, 'images'))
  return [(os.path.join(test_dir, 'images', f), None)
          for f in files if f.endswith('.JPEG')]


def get_image_format(filename):
  """Returns image format from filename."""
  filename = filename.lower()
  if filename.endswith('jpeg') or filename.endswith('jpg'):
    return 'jpeg'
  elif filename.endswith('png'):
    return 'png'
  else:
    raise ValueError('Unrecognized file format: %s' % filename)


class TinyImagenetWriter(object):
  """Helper class which writes Tiny Imagenet dataset into TFRecord file."""

  def __init__(self, tiny_imagenet_wnid_conveter, imagenet_wnid_converter):
    self.tiny_imagenet_wnid_conveter = tiny_imagenet_wnid_conveter
    self.imagenet_wnid_converter = imagenet_wnid_converter

  def write_tf_record(self,
                      annotations,
                      output_file):
    """Generates TFRecord file from given list of annotations."""
    with tf.python_io.TFRecordWriter(output_file) as writer:
      for image_filename, image_metadata in annotations:
        with tf.gfile.Open(image_filename) as f:
          image_buffer = f.read()
        image_format = get_image_format(image_filename)
        features = {
            'image/encoded': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_buffer])),
            'image/format': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_format]))
        }
        if image_metadata:
          # bounding box features
          features['bbox/xmin'] = tf.train.Feature(
              int64_list=tf.train.Int64List(value=[image_metadata.x1]))
          features['bbox/ymin'] = tf.train.Feature(
              int64_list=tf.train.Int64List(value=[image_metadata.y1]))
          features['bbox/xmax'] = tf.train.Feature(
              int64_list=tf.train.Int64List(value=[image_metadata.x2]))
          features['bbox/ymax'] = tf.train.Feature(
              int64_list=tf.train.Int64List(value=[image_metadata.y2]))
          # tiny imagenet label, from [0, 200) iterval
          tiny_imagenet_label = self.tiny_imagenet_wnid_conveter.to_node_id(
              image_metadata.label)
          features['label/wnid'] = tf.train.Feature(
              bytes_list=tf.train.BytesList(value=image_metadata.label))
          features['label/tiny_imagenet'] = tf.train.Feature(
              int64_list=tf.train.Int64List(value=[tiny_imagenet_label]))
          # full imagenet label, from [1, 1001) interval
          if self.imagenet_wnid_converter:
            imagenet_label = self.imagenet_wnid_converter.to_node_id(
                image_metadata.label)
            features['label/imagenet'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[imagenet_label]))
        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())


def main(_):
  assert FLAGS.input_dir, 'Input directory must be provided'
  assert FLAGS.output_dir, 'Output directory must be provided'

  # Create WordNet ID conveters for tiny imagenet and possibly for imagenet
  tiny_imagenet_wnid_conveter = WnIdToNodeIdConverter(
      os.path.join(FLAGS.input_dir, 'wnids.txt'),
      background_class=False)
  if FLAGS.imagenet_synsets_path:
    imagenet_wnid_converter = WnIdToNodeIdConverter(FLAGS.imagenet_synsets_path,
                                                    background_class=True)
  else:
    imagenet_wnid_converter = None

  # read tiny imagenet annotations
  train_annotations = read_training_annotations(
      os.path.join(FLAGS.input_dir, 'train'))
  random.shuffle(train_annotations)
  val_annotations = read_validation_annotations(
      os.path.join(FLAGS.input_dir, 'val'))
  test_filenames = read_test_annotations(os.path.join(FLAGS.input_dir, 'test'))

  # Generate TFRecord files
  writer = TinyImagenetWriter(tiny_imagenet_wnid_conveter,
                              imagenet_wnid_converter)
  tf.logging.info('Converting %d training images', len(train_annotations))
  writer.write_tf_record(train_annotations,
                         os.path.join(FLAGS.output_dir, 'train.tfrecord'))
  tf.logging.info('Converting %d validation images ', len(val_annotations))
  writer.write_tf_record(val_annotations,
                         os.path.join(FLAGS.output_dir, 'validation.tfrecord'))
  tf.logging.info('Converting %d test images', len(test_filenames))
  writer.write_tf_record(test_filenames,
                         os.path.join(FLAGS.output_dir, 'test.tfrecord'))
  tf.logging.info('All files are converted')


if __name__ == '__main__':
  app.run(main)
