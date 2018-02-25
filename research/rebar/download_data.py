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

"""Download MNIST, Omniglot datasets for Rebar."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib
import gzip
import os
import config
import struct
import numpy as np
import cPickle as pickle
import datasets

MNIST_URL = 'see README'
MNIST_BINARIZED_URL = 'see README'
OMNIGLOT_URL = 'see README'

MNIST_FLOAT_TRAIN = 'train-images-idx3-ubyte'


def load_mnist_float(local_filename):
  with open(local_filename, 'rb') as f:
    f.seek(4)
    nimages, rows, cols = struct.unpack('>iii', f.read(12))
    dim = rows*cols

    images = np.fromfile(f, dtype=np.dtype(np.ubyte))
    images = (images/255.0).astype('float32').reshape((nimages, dim))

  return images

if __name__ == '__main__':
  if not os.path.exists(config.DATA_DIR):
    os.makedirs(config.DATA_DIR)

  # Get MNIST and convert to npy file
  local_filename = os.path.join(config.DATA_DIR, MNIST_FLOAT_TRAIN)
  if not os.path.exists(local_filename):
    urllib.urlretrieve("%s/%s.gz" % (MNIST_URL, MNIST_FLOAT_TRAIN), local_filename+'.gz')
    with gzip.open(local_filename+'.gz', 'rb') as f:
      file_content = f.read()
    with open(local_filename, 'wb') as f:
      f.write(file_content)
    os.remove(local_filename+'.gz')

  mnist_float_train = load_mnist_float(local_filename)[:-10000]
  # save in a nice format
  np.save(os.path.join(config.DATA_DIR, config.MNIST_FLOAT), mnist_float_train)

  # Get binarized MNIST
  splits = ['train', 'valid', 'test']
  mnist_binarized = []
  for split in splits:
    filename = 'binarized_mnist_%s.amat' % split
    url = '%s/binarized_mnist_%s.amat' % (MNIST_BINARIZED_URL, split)
    local_filename = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(local_filename):
      urllib.urlretrieve(url, local_filename)

    with open(local_filename, 'rb') as f:
      mnist_binarized.append((np.array([map(int, line.split()) for line in f.readlines()]).astype('float32'), None))

  # save in a nice format
  with open(os.path.join(config.DATA_DIR, config.MNIST_BINARIZED), 'w') as out:
    pickle.dump(mnist_binarized, out)

  # Get Omniglot
  local_filename = os.path.join(config.DATA_DIR, config.OMNIGLOT)
  if not os.path.exists(local_filename):
    urllib.urlretrieve(OMNIGLOT_URL,
                       local_filename)

