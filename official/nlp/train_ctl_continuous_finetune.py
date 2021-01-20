# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""TFM continuous finetuning+eval training driver."""
from absl import app
from absl import flags
import gin

# pylint: disable=unused-import
from official.common import registry_imports
# pylint: enable=unused-import
from official.common import flags as tfm_flags
from official.core import train_utils
from official.nlp import continuous_finetune_lib

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'pretrain_steps',
    default=None,
    help='The number of total training steps for the pretraining job.')


def main(_):
  # TODO(b/177863554): consolidate to nlp/train.py
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir
  train_utils.serialize_config(params, model_dir)
  continuous_finetune_lib.run_continuous_finetune(FLAGS.mode, params, model_dir,
                                                  FLAGS.pretrain_steps)


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
