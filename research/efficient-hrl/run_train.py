# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

r"""Script for training an RL agent using the UVF algorithm.

To run locally: See scripts/local_train.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import gin.tf
# pylint: enable=unused-import
import train
# pylint: disable=unused-import

flags = tf.app.flags
FLAGS = flags.FLAGS


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.config_file:
    for config_file in FLAGS.config_file:
      gin.parse_config_file(config_file)
  if FLAGS.params:
    gin.parse_config(FLAGS.params)

  assert FLAGS.train_dir, "Flag 'train_dir' must be set."
  return train.train_uvf(FLAGS.train_dir)


if __name__ == '__main__':
  tf.app.run()
