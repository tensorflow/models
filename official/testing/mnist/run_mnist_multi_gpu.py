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
Downloads data and runs MNIST model on multi-gpu.
TODO(karmel): This should live in end_to_end testing dir when that is avail.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess


MODEL_DIR = 'official/mnist'
MODEL_MAIN_SCRIPT = 'mnist_multi_gpu.py'


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-r', '--repo_loc', type=str, default='', required=True,
                      help='Full path to local Models repo.'
                           'Should be parent of `official` dir.')
  parser.add_argument('--run_file', type=str, default='',
                      help='MNIST main file to run. Default: '
                      + MODEL_MAIN_SCRIPT)

  args, unparsed = parser.parse_known_args()
  path = os.path.join(args.repo_loc, MODEL_DIR)

  # Run the main training loop
  run_file = args.run_file or MODEL_MAIN_SCRIPT
  main_exec = os.path.join(path, MODEL_MAIN_SCRIPT)
  unparsed_args = ' '.join(unparsed)
  main_cmd = 'python {} {}'.format(main_exec, unparsed_args)
  subprocess.call(main_cmd, shell=True)
