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
"""Common utils for tests for object detection tflearn model."""

from __future__ import absolute_import

import os
import tempfile
import tensorflow as tf


from object_detection import model
from object_detection import model_hparams

FLAGS = tf.flags.FLAGS

FASTER_RCNN_MODEL_NAME = 'faster_rcnn_resnet50_pets'
SSD_INCEPTION_MODEL_NAME = 'ssd_inception_v2_pets'
PATH_BASE = 'google3/third_party/tensorflow_models/object_detection/'


def GetPipelineConfigPath(model_name):
  """Returns path to the local pipeline config file."""
  return os.path.join(FLAGS.test_srcdir, PATH_BASE, 'samples', 'configs',
                      model_name + '.config')


def InitializeFlags(model_name_for_test):
  FLAGS.model_dir = tempfile.mkdtemp()
  FLAGS.pipeline_config_path = GetPipelineConfigPath(model_name_for_test)


def BuildExperiment():
  """Builds an Experiment object for testing purposes."""
  run_config = tf.contrib.learn.RunConfig()
  hparams = model_hparams.create_hparams(
      hparams_overrides='load_pretrained=false')

  # pylint: disable=protected-access
  experiment_fn = model.build_experiment_fn(10, 10)
  # pylint: enable=protected-access
  return experiment_fn(run_config, hparams)
