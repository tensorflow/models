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

from datetime import datetime
import json
import logging
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


def prepare_dirs_and_logger(config):
  formatter = logging.Formatter('%(asctime)s:%(levelname)s::%(message)s')
  logger = logging.getLogger('tensorflow')

  for hdlr in logger.handlers:
    logger.removeHandler(hdlr)

  handler = logging.StreamHandler()
  handler.setFormatter(formatter)

  logger.addHandler(handler)
  logger.setLevel(tf.logging.INFO)

  config.log_dir = os.path.join(config.exp_dir, config.log_dir,
                                config.train_tag)
  config.model_dir = os.path.join(config.exp_dir, config.model_dir,
                                  config.train_tag)
  config.output_dir = os.path.join(config.exp_dir, config.output_dir,
                                   config.train_tag)

  for path in [
    config.log_dir, config.model_dir, config.output_dir
  ]:
    if not os.path.exists(path):
      os.makedirs(path)

  config.data_files = {
    'train': os.path.join(config.data_dir, config.train_data_file),
    'dev': os.path.join(config.data_dir, config.dev_data_file),
    'test': os.path.join(config.data_dir, config.test_data_file)
  }

  return config


def get_time():
  return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_config(config):
  param_path = os.path.join(config.model_dir, 'params.json')

  tf.logging.info('log dir: %s' % config.log_dir)
  tf.logging.info('model dir: %s' % config.model_dir)
  tf.logging.info('param path: %s' % param_path)
  tf.logging.info('output dir: %s' % config.output_dir)

  with open(param_path, 'w') as f:
    f.write(json.dumps(config.__dict__, indent=4, sort_keys=True))
