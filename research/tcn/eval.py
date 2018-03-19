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

"""Calculates running validation of TCN models (and baseline comparisons)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from estimators.get_estimator import get_estimator
from utils import util
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string(
    'config_paths', '',
    """
    Path to a YAML configuration files defining FLAG values. Multiple files
    can be separated by the `#` symbol. Files are merged recursively. Setting
    a key in these files is equivalent to setting the FLAG value with
    the same name.
    """)
tf.flags.DEFINE_string(
    'model_params', '{}', 'YAML configuration string for the model parameters.')
tf.app.flags.DEFINE_string('master', 'local',
                           'BNS name of the TensorFlow master to use')
tf.app.flags.DEFINE_string(
    'logdir', '/tmp/tcn', 'Directory where to write event logs.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  """Runs main eval loop."""
  # Parse config dict from yaml config files / command line flags.
  logdir = FLAGS.logdir
  config = util.ParseConfigsToLuaTable(FLAGS.config_paths, FLAGS.model_params)

  # Choose an estimator based on training strategy.
  estimator = get_estimator(config, logdir)

  # Wait for the first checkpoint file to be written.
  while not tf.train.latest_checkpoint(logdir):
    tf.logging.info('Waiting for a checkpoint file...')
    time.sleep(10)

  # Run validation.
  while True:
    estimator.evaluate()

if __name__ == '__main__':
  tf.app.run()
