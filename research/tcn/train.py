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

"""Trains TCN models (and baseline comparisons)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')
tf.app.flags.DEFINE_integer(
    'ps_tasks', 0, 'Number of tasks in the ps job. If 0 no ps job is used.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  """Runs main training loop."""
  # Parse config dict from yaml config files / command line flags.
  config = util.ParseConfigsToLuaTable(
      FLAGS.config_paths, FLAGS.model_params, save=True, logdir=FLAGS.logdir)

  # Choose an estimator based on training strategy.
  estimator = get_estimator(config, FLAGS.logdir)

  # Run training
  estimator.train()

if __name__ == '__main__':
  tf.app.run()
