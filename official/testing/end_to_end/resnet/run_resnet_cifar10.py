# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""
Downloads data and runs ResNet model on the CIFAR-10 dataset.

TODO(karmel): Clean this up and make reusable.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess


MODEL_DIR = 'official/resnet'
DATA_EXTRACTION_SCRIPT = 'cifar10_download_and_extract.py'
MODEL_MAIN_SCRIPT = 'cifar10_main.py'


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('-r', '--repo_loc', type=str, default='', required=True,
    help='Full path to local Models repo. Should be parent of `official` dir.')
  parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
    help='The location where data will be downloaded to and accessed.')


  args, unparsed = parser.parse_known_args()
  path = os.path.join(args.repo_loc, MODEL_DIR)

  # Download data.
  # TODO(karmel): Data dir cleaning and processing?
  data_exec = os.path.join(path, DATA_EXTRACTION_SCRIPT)
  data_cmd = "python {} --data_dir={}".format(data_exec, args.data_dir)
  subprocess.call(data_cmd, shell=True)

  # Run
  main_exec = os.path.join(path, MODEL_MAIN_SCRIPT)
  unparsed_args = ' '.join(unparsed)
  main_cmd = "python {} --data_dir={} {}".format(
    main_exec, args.data_dir, unparsed_args)
  subprocess.call(main_cmd, shell=True)
