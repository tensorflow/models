# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Utilities for tensorboard."""

from tensorboard import program

from delf.python.training import global_features_utils


def launch_tensorboard(log_dir):
  """Runs tensorboard with the given `log_dir`.

  Args:
    log_dir: String, directory to launch tensorboard in.
  """
  tensorboard = program.TensorBoard()
  tensorboard.configure(argv=[None, '--logdir', log_dir])
  url = tensorboard.launch()
  global_features_utils.debug_and_log("Launching Tensorboard: {}".format(url))
