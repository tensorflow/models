# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts cifar10/100 data to TFRecords of TF-Example protos.

This module downloads the cifar data, uncompresses it, reads the files that
make up the cifar data and creates two TFRecord datasets: one for train and one
for test. Each TFRecord dataset is comprised of a set of TF-Example protocol
buffers, each of which contain a single image and label.

The script should take several minutes to run.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle
import os
import six
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from datasets import dataset_utils

# The URLs where the CIFAR data can be downloaded.
_DATA_URL = {
  'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
  'cifar100': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
}

_DATA_DIR = {
  'cifar10': 'cifar-10-batches-py',
  'cifar100': 'cifar-100-python',
}

# The number of training files.
_NUM_TRAIN_FILES = {
  'cifar10': 5,
  'cifar100': 1,
}

# The height and width of each image.
_IMAGE_SIZE = 32

# The names of the classes.
_CLASS_NAMES = {
  'cifar10': [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
  ],
  'cifar100': [
  'apple',
  'aquarium_fish',
  'baby',
  'bear',
  'beaver',
  'bed',
  'bee',
  'beetle',
  'bicycle',
  'bottle',
  'bowl',
  'boy',
  'bridge',
  'bus',
  'butterfly',
  'camel',
  'can',
  'castle',
  'caterpillar',
  'cattle',
  'chair',
  'chimpanzee',
  'clock',
  'cloud',
  'cockroach',
  'couch',
  'crab',
  'crocodile',
  'cup',
  'dinosaur',
  'dolphin',
  'elephant',
  'flatfish',
  'forest',
  'fox',
  'girl',
  'hamster',
  'house',
  'kangaroo',
  'keyboard',
  'lamp',
  'lawn_mower',
  'leopard',
  'lion',
  'lizard',
  'lobster',
  'man',
  'maple_tree',
  'motorcycle',
  'mountain',
  'mouse',
  'mushroom',
  'oak_tree',
  'orange',
  'orchid',
  'otter',
  'palm_tree',
  'pear',
  'pickup_truck',
  'pine_tree',
  'plain',
  'plate',
  'poppy',
  'porcupine',
  'possum',
  'rabbit',
  'raccoon',
  'ray',
  'road',
  'rocket',
  'rose',
  'sea',
  'seal',
  'shark',
  'shrew',
  'skunk',
  'skyscraper',
  'snail',
  'snake',
  'spider',
  'squirrel',
  'streetcar',
  'sunflower',
  'sweet_pepper',
  'table',
  'tank',
  'telephone',
  'television',
  'tiger',
  'tractor',
  'train',
  'trout',
  'tulip',
  'turtle',
  'wardrobe',
  'whale',
  'willow_tree',
  'wolf',
  'woman',
  'worm',
  ]
}

_COARSE_CLASS_NAMES = [
  'aquatic_mammals',
  'fish',
  'flowers',
  'food_containers',
  'fruit_and_vegetables',
  'household_electrical_devices',
  'household_furniture',
  'insects',
  'large_carnivores',
  'large_man-made_outdoor_things',
  'large_natural_outdoor_scenes',
  'large_omnivores_and_herbivores',
  'medium_mammals',
  'non-insect_invertebrates',
  'people',
  'reptiles',
  'small_mammals',
  'trees',
  'vehicles_1',
  'vehicles_2',
]

_COARSE_LABELS_FILENAME = 'coarse_labels.txt'

def _add_to_tfrecord(filename, tfrecord_writer, dataset, offset=0):
  """Loads data from the cifar pickle files and writes files to a TFRecord.

  Args:
    filename: The filename of the cifar pickle file.
    tfrecord_writer: The TFRecord writer to use for writing.
    offset: An offset into the absolute number of images previously written.

  Returns:
    The new offset.
  """
  with open(filename, 'rb') as f:
    if six.PY3:
        data = cPickle.load(f, encoding='bytes')
    else:
        data = cPickle.load(f)

  images = data[six.b('data')]
  num_images = images.shape[0]

  images = images.reshape((num_images, 3, 32, 32))
  labels = data[six.b('labels' if dataset == 'cifar10' else 'fine_labels')]
  if dataset == 'cifar100':
      coarse_labels = data[six.b('coarse_labels')]

  with tf.Graph().as_default():
    image_placeholder = tf.placeholder(dtype=tf.uint8)
    encoded_image = tf.image.encode_png(image_placeholder)

    with tf.Session('') as sess:

      for j in range(num_images):
        sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
            filename, offset + j + 1, offset + num_images))
        sys.stdout.flush()

        image = np.squeeze(images[j]).transpose((1, 2, 0))
        label = labels[j]
        coarse_label = coarse_labels[j] if dataset == 'cifar100' else None

        png_string = sess.run(encoded_image,
                              feed_dict={image_placeholder: image})

        example = dataset_utils.image_to_tfexample(
            png_string, six.b('png'),
            _IMAGE_SIZE, _IMAGE_SIZE,
            label, coarse_class_id = coarse_label)
        tfrecord_writer.write(example.SerializeToString())

  return offset + num_images


def _get_output_filename(dataset_dir, split_name, dataset):
  """Creates the output filename.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    split_name: The name of the train/test split.

  Returns:
    An absolute file path.
  """
  return '%s/%s_%s.tfrecord' % (dataset_dir, dataset, split_name)


def _download_and_uncompress_dataset(dataset_dir, dataset):
  """Downloads cifar and uncompresses it locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL[dataset].split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(_DATA_URL[dataset],
                                             filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def _clean_up_temporary_files(dataset_dir, dataset):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL[dataset].split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, _DATA_DIR[dataset])
  tf.gfile.DeleteRecursively(tmp_dir)

def _batch_name(split_name, offset, dataset):
  """Returns the file name for a given batch.
  
  Args:
    split_name: "train"|"test"
    offset: batch index (0-indexed)
    dataset: "cifar10"|"cifar100"
  """
  if dataset == 'cifar10':
    if split_name == 'train':
      return 'data_batch_%d' % (offset + 1)  # 1-indexed.
    else:
      return 'test_batch'
  else:
    return split_name

def run(dataset_dir, dataset):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  training_filename = _get_output_filename(dataset_dir, 'train',
                                           dataset=dataset)
  testing_filename = _get_output_filename(dataset_dir, 'test',
                                          dataset=dataset)

  if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download_and_uncompress_tarball(_DATA_URL[dataset], dataset_dir)

  # First, process the training data:
  with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
    offset = 0
    for i in range(_NUM_TRAIN_FILES[dataset]):
      filename = os.path.join(dataset_dir,
                              _DATA_DIR[dataset],
                              _batch_name('train', offset=i, dataset=dataset))
      offset = _add_to_tfrecord(filename, tfrecord_writer, dataset, offset)

  # Next, process the testing data:
  with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    filename = os.path.join(dataset_dir,
                            _DATA_DIR[dataset],
                            _batch_name('test', offset=0, dataset=dataset))
    _add_to_tfrecord(filename, tfrecord_writer, dataset)

  # Finally, write the labels file:
  labels_to_class_names = dict(enumerate(_CLASS_NAMES[dataset]))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  if dataset == 'cifar100':
    coarse_labels_to_class_names = dict(enumerate(_COARSE_CLASS_NAMES))
    dataset_utils.write_label_file(coarse_labels_to_class_names, dataset_dir,
                                   filename=_COARSE_LABELS_FILENAME)
    

  _clean_up_temporary_files(dataset_dir, dataset)
  print('\nFinished converting the %s dataset!' % dataset)
