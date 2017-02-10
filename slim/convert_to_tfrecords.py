# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Converts a directory of images to TFRecords of TF-Example protos.
Images belonging to a class must be in their own directory organized as show below. Each class can consist of any number
 of images, image sizes, and names aslong as they are JPEG and adhear to the file system. Images are encoded in their
 original size and must be preprocessed before using.
File System:
Root DIR --> Class 1 --> img1.jpg
 --> image2.jpg
 --> Class 2 --> someimage.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

_TRAIN_DIR = '/home/milab/Pictures/Preliminary_Cellphone_Database/Train/'
_VALIDATION_DIR = '/home/milab/Pictures/Preliminary_Cellphone_Database/Test/'
_DEST_DIR = '/tmp/cellphone/'

_FILE_PATTERN = 'cellphone_%s_%05d-of-%05d.tfrecord'

# The number of shards per dataset split.
_NUM_SHARDS = 2


class ImageReader(object):
 """Helper class that provides TensorFlow image coding utilities."""

 def __init__(self):
  # Initializes function that decodes RGB JPEG data.
  self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
  self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

 def read_image_dims(self, sess, image_data):
  image = self.decode_jpeg(sess, image_data)
  return image.shape[0], image.shape[1]

 def decode_jpeg(self, sess, image_data):
  image = sess.run(self._decode_jpeg,
  feed_dict={self._decode_jpeg_data: image_data})
  assert len(image.shape) == 3
  assert image.shape[2] == 3
  return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.
  Args:
  dataset_dir: A directory containing a set of subdirectories representing
  class names. Each subdirectory should contain PNG or JPG encoded images.
  Returns:
  A list of image file paths, relative to `dataset_dir` and the list of
  subdirectories, representing class names.
  """
  directories = []
  class_names = []
  for filename in os.listdir(dataset_dir):
  path = os.path.join(dataset_dir, filename)
  if os.path.isdir(path):
  directories.append(path)
  class_names.append(filename)

  photo_filenames = []
  for directory in directories:
  for filename in os.listdir(directory):
  path = os.path.join(directory, filename)
  photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
 output_filename = _FILE_PATTERN % (
 split_name, shard_id, _NUM_SHARDS)
 return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
 """Converts the given filenames to a TFRecord dataset.
 Args:
 split_name: The name of the dataset, either 'train' or 'validation'.
 filenames: A list of absolute paths to png or jpg images.
 class_names_to_ids: A dictionary from class names (strings) to ids
 (integers).
 dataset_dir: The directory where the converted datasets are stored.
 """
 assert split_name in ['train', 'validation']

 num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

 with tf.Graph().as_default():
 image_reader = ImageReader()

 with tf.Session('') as sess:

 for shard_id in range(_NUM_SHARDS):
 output_filename = _get_dataset_filename(
 dataset_dir, split_name, shard_id)

 with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
 start_ndx = shard_id * num_per_shard
 end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
 for i in range(start_ndx, end_ndx):
 sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
 i + 1, len(filenames), shard_id))
 sys.stdout.flush()

 # Read the filename:
 image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
 height, width = image_reader.read_image_dims(sess, image_data)

 class_name = os.path.basename(os.path.dirname(filenames[i]))
 class_id = class_names_to_ids[class_name]

 example = dataset_utils.image_to_tfexample(
 image_data, 'jpg', height, width, class_id)
 tfrecord_writer.write(example.SerializeToString())

 sys.stdout.write('\n')
 sys.stdout.flush()


def _dataset_exists(dataset_dir):
 for split_name in ['train', 'validation']:
 for shard_id in range(_NUM_SHARDS):
 output_filename = _get_dataset_filename(
 dataset_dir, split_name, shard_id)
 if not tf.gfile.Exists(output_filename):
 return False
 return True


def main(_):
 """Runs the download and conversion operation.
 Args:
 dataset_dir: The dataset directory where the dataset is stored.
 """
 if not tf.gfile.Exists(_DEST_DIR):
 tf.gfile.MakeDirs(_DEST_DIR)

 # Get train and test filenames and class IDs
 training_filenames, class_names = _get_filenames_and_classes(_TRAIN_DIR)
 validation_filenames, class_names = _get_filenames_and_classes(_VALIDATION_DIR)
 class_names_to_ids = dict(zip(class_names, range(len(class_names))))

 # First, convert the training and validation sets.
 _convert_dataset('train', training_filenames, class_names_to_ids,
 _DEST_DIR)
 _convert_dataset('validation', validation_filenames, class_names_to_ids,
 _DEST_DIR)

 # Finally, write the labels file:
 labels_to_class_names = dict(zip(range(len(class_names)), class_names))
 dataset_utils.write_label_file(labels_to_class_names, _DEST_DIR)

 print('\nFinished converting the dataset!')


if __name__ == '__main__':
 tf.app.run()