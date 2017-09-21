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

"""Helper functions used in both train and inference."""

import json
import os.path

import tensorflow as tf


def GetConfigString(config_file):
  config_string = ''
  if config_file is not None:
    config_string = open(config_file).read()
  return config_string


class InputConfig(object):

  def __init__(self, config_string):
    config = json.loads(config_string)
    self.data = config["data"]
    self.unique_code_size = config["unique_code_size"]


class TrainConfig(object):

  def __init__(self, config_string):
    config = json.loads(config_string)
    self.batch_size = config["batch_size"]
    self.learning_rate = config["learning_rate"]
    self.decay_rate = config["decay_rate"]
    self.samples_per_decay = config["samples_per_decay"]


def SaveConfig(directory, filename, config_string):
  path = os.path.join(directory, filename)
  with tf.gfile.Open(path, mode='w') as f:
    f.write(config_string)
