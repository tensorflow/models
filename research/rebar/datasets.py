# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Library of datasets for REBAR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import scipy.io
import numpy as np
import cPickle as pickle
import tensorflow as tf
import config
gfile = tf.gfile


def load_data(hparams):
  # Load data
  if hparams.task in ['sbn', 'sp']:
    reader = read_MNIST
  elif hparams.task == 'omni':
    reader = read_omniglot
  x_train, x_valid, x_test = reader(binarize=not hparams.dynamic_b)

  return x_train, x_valid, x_test

def read_MNIST(binarize=False):
  """Reads in MNIST images.

  Args:
    binarize: whether to use the fixed binarization

  Returns:
    x_train: 50k training images
    x_valid: 10k validation images
    x_test: 10k test images

  """
  with gfile.FastGFile(os.path.join(config.DATA_DIR, config.MNIST_BINARIZED), 'r') as f:
    (x_train, _), (x_valid, _), (x_test, _) = pickle.load(f)

  if not binarize:
    with gfile.FastGFile(os.path.join(config.DATA_DIR, config.MNIST_FLOAT), 'r') as f:
      x_train = np.load(f).reshape(-1, 784)

  return x_train, x_valid, x_test

def read_omniglot(binarize=False):
  """Reads in Omniglot images.

  Args:
    binarize: whether to use the fixed binarization

  Returns:
    x_train: training images
    x_valid: validation images
    x_test: test images

  """
  n_validation=1345

  def reshape_data(data):
    return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

  omni_raw = scipy.io.loadmat(os.path.join(config.DATA_DIR, config.OMNIGLOT))

  train_data = reshape_data(omni_raw['data'].T.astype('float32'))
  test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

  # Binarize the data with a fixed seed
  if binarize:
    np.random.seed(5)
    train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
    test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

  shuffle_seed = 123
  permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
  train_data = train_data[permutation]

  x_train = train_data[:-n_validation]
  x_valid = train_data[-n_validation:]
  x_test = test_data

  return x_train, x_valid, x_test

