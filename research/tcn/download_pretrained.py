# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
"""Downloads pretrained InceptionV3 and ResnetV2-50 checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import urllib

INCEPTION_URL = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz'
RESNET_URL = 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz'


def DownloadWeights(model_dir, url):
  os.makedirs(model_dir)
  tar_path = os.path.join(model_dir, 'ckpt.tar.gz')
  urllib.urlretrieve(url, tar_path)
  tar = tarfile.open(os.path.join(model_dir, 'ckpt.tar.gz'))
  tar.extractall(model_dir)


if __name__ == '__main__':

  # Create a directory for all pretrained checkpoints.
  ckpt_dir = 'pretrained_checkpoints'
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  # Download inception.
  print('Downloading inception pretrained weights...')
  inception_dir = os.path.join(ckpt_dir, 'inception')
  DownloadWeights(inception_dir, INCEPTION_URL)
  print('Done downloading inception pretrained weights.')

  print('Downloading resnet pretrained weights...')
  resnet_dir = os.path.join(ckpt_dir, 'resnet')
  DownloadWeights(resnet_dir, RESNET_URL)
  print('Done downloading resnet pretrained weights.')

