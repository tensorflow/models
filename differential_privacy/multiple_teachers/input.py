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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import gzip
import math
import numpy as np
import os
from scipy.io import loadmat as loadmat
from six.moves import urllib
import sys
import tarfile

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def create_dir_if_needed(dest_directory):
  """
  Create directory if doesn't exist
  :param dest_directory:
  :return: True if everything went well
  """
  if not tf.gfile.IsDirectory(dest_directory):
    tf.gfile.MakeDirs(dest_directory)

  return True


def maybe_download(file_urls, directory):
  """
  Download a set of files in temporary local folder
  :param directory: the directory where to download 
  :return: a tuple of filepaths corresponding to the files given as input
  """
  # Create directory if doesn't exist
  assert create_dir_if_needed(directory)

  # This list will include all URLS of the local copy of downloaded files
  result = []

  # For each file of the dataset
  for file_url in file_urls:
    # Extract filename
    filename = file_url.split('/')[-1]

    # If downloading from GitHub, remove suffix ?raw=True from local filename
    if filename.endswith("?raw=true"):
      filename = filename[:-9]

    # Deduce local file url
    #filepath = os.path.join(directory, filename)
    filepath = directory + '/' + filename

    # Add to result list
    result.append(filepath)

    # Test if file already exists
    if not gfile.Exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(file_url, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  return result


def image_whitening(data):
  """
  Subtracts mean of image and divides by adjusted standard variance (for
  stability). Operations are per image but performed for the entire array.
  :param image: 4D array (ID, Height, Weight, Channel)
  :return: 4D array (ID, Height, Weight, Channel)
  """
  assert len(np.shape(data)) == 4

  # Compute number of pixels in image
  nb_pixels = np.shape(data)[1] * np.shape(data)[2] * np.shape(data)[3]

  # Subtract mean
  mean = np.mean(data, axis=(1,2,3))

  ones = np.ones(np.shape(data)[1:4], dtype=np.float32)
  for i in xrange(len(data)):
    data[i, :, :, :] -= mean[i] * ones

  # Compute adjusted standard variance
  adj_std_var = np.maximum(np.ones(len(data), dtype=np.float32) / math.sqrt(nb_pixels), np.std(data, axis=(1,2,3))) #NOLINT(long-line)

  # Divide image
  for i in xrange(len(data)):
    data[i, :, :, :] = data[i, :, :, :] / adj_std_var[i]

  print(np.shape(data))

  return data


def extract_svhn(local_url):
  """
  Extract a MATLAB matrix into two numpy arrays with data and labels
  :param local_url:
  :return:
  """

  with gfile.Open(local_url, mode='r') as file_obj:
    # Load MATLAB matrix using scipy IO
    dict = loadmat(file_obj)

    # Extract each dictionary (one for data, one for labels)
    data, labels = dict["X"], dict["y"]

    # Set np type
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    # Transpose data to match TF model input format
    data = data.transpose(3, 0, 1, 2)

    # Fix the SVHN labels which label 0s as 10s
    labels[labels == 10] = 0

    # Fix label dimensions
    labels = labels.reshape(len(labels))

    return data, labels


def unpickle_cifar_dic(file):
  """
  Helper function: unpickles a dictionary (used for loading CIFAR)
  :param file: filename of the pickle
  :return: tuple of (images, labels)
  """
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict['data'], dict['labels']


def extract_cifar10(local_url, data_dir):
  """
  Extracts the CIFAR-10 dataset and return numpy arrays with the different sets
  :param local_url: where the tar.gz archive is located locally
  :param data_dir: where to extract the archive's file
  :return: a tuple (train data, train labels, test data, test labels)
  """
  # These numpy dumps can be reloaded to avoid performing the pre-processing
  # if they exist in the working directory.
  # Changing the order of this list will ruin the indices below.
  preprocessed_files = ['/cifar10_train.npy',
                        '/cifar10_train_labels.npy',
                        '/cifar10_test.npy',
                        '/cifar10_test_labels.npy']

  all_preprocessed = True
  for file in preprocessed_files:
    if not tf.gfile.Exists(data_dir + file):
      all_preprocessed = False
      break

  if all_preprocessed:
    # Reload pre-processed training data from numpy dumps
    with tf.gfile.Open(data_dir + preprocessed_files[0], mode='r') as file_obj:
      train_data = np.load(file_obj)
    with tf.gfile.Open(data_dir + preprocessed_files[1], mode='r') as file_obj:
      train_labels = np.load(file_obj)

    # Reload pre-processed testing data from numpy dumps
    with tf.gfile.Open(data_dir + preprocessed_files[2], mode='r') as file_obj:
      test_data = np.load(file_obj)
    with tf.gfile.Open(data_dir + preprocessed_files[3], mode='r') as file_obj:
      test_labels = np.load(file_obj)

  else:
    # Do everything from scratch
    # Define lists of all files we should extract
    train_files = ["data_batch_" + str(i) for i in xrange(1,6)]
    test_file = ["test_batch"]
    cifar10_files = train_files + test_file

    # Check if all files have already been extracted
    need_to_unpack = False
    for file in cifar10_files:
      if not tf.gfile.Exists(file):
        need_to_unpack = True
        break

    # We have to unpack the archive
    if need_to_unpack:
      tarfile.open(local_url, 'r:gz').extractall(data_dir)

    # Load training images and labels
    images = []
    labels = []
    for file in train_files:
      # Construct filename
      filename = data_dir + "/cifar-10-batches-py/" + file

      # Unpickle dictionary and extract images and labels
      images_tmp, labels_tmp = unpickle_cifar_dic(filename)

      # Append to lists
      images.append(images_tmp)
      labels.append(labels_tmp)

    # Convert to numpy arrays and reshape in the expected format
    train_data = np.asarray(images, dtype=np.float32).reshape((50000,3,32,32))
    train_data = np.swapaxes(train_data, 1, 3)
    train_labels = np.asarray(labels, dtype=np.int32).reshape(50000)

    # Save so we don't have to do this again
    np.save(data_dir + preprocessed_files[0], train_data)
    np.save(data_dir + preprocessed_files[1], train_labels)

    # Construct filename for test file
    filename = data_dir + "/cifar-10-batches-py/" + test_file[0]

    # Load test images and labels
    test_data, test_images = unpickle_cifar_dic(filename)

    # Convert to numpy arrays and reshape in the expected format
    test_data = np.asarray(test_data,dtype=np.float32).reshape((10000,3,32,32))
    test_data = np.swapaxes(test_data, 1, 3)
    test_labels = np.asarray(test_images, dtype=np.int32).reshape(10000)

    # Save so we don't have to do this again
    np.save(data_dir + preprocessed_files[2], test_data)
    np.save(data_dir + preprocessed_files[3], test_labels)

  return train_data, train_labels, test_data, test_labels


def extract_mnist_data(filename, num_images, image_size, pixel_depth):
  """
  Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  # if not os.path.exists(file):
  if not tf.gfile.Exists(filename+".npy"):
    with gzip.open(filename) as bytestream:
      bytestream.read(16)
      buf = bytestream.read(image_size * image_size * num_images)
      data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
      data = (data - (pixel_depth / 2.0)) / pixel_depth
      data = data.reshape(num_images, image_size, image_size, 1)
      np.save(filename, data)
      return data
  else:
    with tf.gfile.Open(filename+".npy", mode='r') as file_obj:
      return np.load(file_obj)


def extract_mnist_labels(filename, num_images):
  """
  Extract the labels into a vector of int64 label IDs.
  """
  # if not os.path.exists(file):
  if not tf.gfile.Exists(filename+".npy"):
    with gzip.open(filename) as bytestream:
      bytestream.read(8)
      buf = bytestream.read(1 * num_images)
      labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
      np.save(filename, labels)
    return labels
  else:
    with tf.gfile.Open(filename+".npy", mode='r') as file_obj:
      return np.load(file_obj)


def ld_svhn(extended=False, test_only=False):
  """
  Load the original SVHN data
  :param extended: include extended training data in the returned array
  :param test_only: disables loading of both train and extra -> large speed up
  :return: tuple of arrays which depend on the parameters
  """
  # Define files to be downloaded
  # WARNING: changing the order of this list will break indices (cf. below)
  file_urls = ['http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
               'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
               'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat']

  # Maybe download data and retrieve local storage urls
  local_urls = maybe_download(file_urls, FLAGS.data_dir)

  # Extra Train, Test, and Extended Train data
  if not test_only:
    # Load and applying whitening to train data
    train_data, train_labels = extract_svhn(local_urls[0])
    train_data = image_whitening(train_data)

    # Load and applying whitening to extended train data
    ext_data, ext_labels = extract_svhn(local_urls[2])
    ext_data = image_whitening(ext_data)

  # Load and applying whitening to test data
  test_data, test_labels = extract_svhn(local_urls[1])
  test_data = image_whitening(test_data)

  if test_only:
    return test_data, test_labels
  else:
    if extended:
      # Stack train data with the extended training data
      train_data = np.vstack((train_data, ext_data))
      train_labels = np.hstack((train_labels, ext_labels))

      return train_data, train_labels, test_data, test_labels
    else:
      # Return training and extended training data separately
      return train_data,train_labels, test_data,test_labels, ext_data,ext_labels


def ld_cifar10(test_only=False):
  """
  Load the original CIFAR10 data
  :param extended: include extended training data in the returned array
  :param test_only: disables loading of both train and extra -> large speed up
  :return: tuple of arrays which depend on the parameters
  """
  # Define files to be downloaded
  file_urls = ['https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']

  # Maybe download data and retrieve local storage urls
  local_urls = maybe_download(file_urls, FLAGS.data_dir)

  # Extract archives and return different sets
  dataset = extract_cifar10(local_urls[0], FLAGS.data_dir)

  # Unpack tuple
  train_data, train_labels, test_data, test_labels = dataset

  # Apply whitening to input data
  train_data = image_whitening(train_data)
  test_data = image_whitening(test_data)

  if test_only:
    return test_data, test_labels
  else:
    return train_data, train_labels, test_data, test_labels


def ld_mnist(test_only=False):
  """
  Load the MNIST dataset
  :param extended: include extended training data in the returned array
  :param test_only: disables loading of both train and extra -> large speed up
  :return: tuple of arrays which depend on the parameters
  """
  # Define files to be downloaded
  # WARNING: changing the order of this list will break indices (cf. below)
  file_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
               'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
               'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
               'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
               ]

  # Maybe download data and retrieve local storage urls
  local_urls = maybe_download(file_urls, FLAGS.data_dir)

  # Extract it into np arrays.
  train_data = extract_mnist_data(local_urls[0], 60000, 28, 1)
  train_labels = extract_mnist_labels(local_urls[1], 60000)
  test_data = extract_mnist_data(local_urls[2], 10000, 28, 1)
  test_labels = extract_mnist_labels(local_urls[3], 10000)

  if test_only:
    return test_data, test_labels
  else:
    return train_data, train_labels, test_data, test_labels


def partition_dataset(data, labels, nb_teachers, teacher_id):
  """
  Simple partitioning algorithm that returns the right portion of the data
  needed by a given teacher out of a certain nb of teachers
  :param data: input data to be partitioned
  :param labels: output data to be partitioned
  :param nb_teachers: number of teachers in the ensemble (affects size of each
                      partition)
  :param teacher_id: id of partition to retrieve
  :return:
  """

  # Sanity check
  assert len(data) == len(labels)
  assert int(teacher_id) < int(nb_teachers)

  # This will floor the possible number of batches
  batch_len = int(len(data) / nb_teachers)

  # Compute start, end indices of partition
  start = teacher_id * batch_len
  end = (teacher_id+1) * batch_len

  # Slice partition off
  partition_data = data[start:end]
  partition_labels = labels[start:end]

  return partition_data, partition_labels
