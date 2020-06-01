# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""ALBERT classification finetuning runner in tf2.x."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.albert import configs as albert_configs
from official.nlp.bert import run_classifier as run_classifier_bert
from official.utils.misc import distribution_utils

FLAGS = flags.FLAGS


def main(_):
  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  max_seq_length = input_meta_data['max_seq_length']
  train_input_fn = run_classifier_bert.get_dataset_fn(
      FLAGS.train_data_path,
      max_seq_length,
      FLAGS.train_batch_size,
      is_training=True)
  eval_input_fn = run_classifier_bert.get_dataset_fn(
      FLAGS.eval_data_path,
      max_seq_length,
      FLAGS.eval_batch_size,
      is_training=False)

  albert_config = albert_configs.AlbertConfig.from_json_file(
      FLAGS.bert_config_file)
  run_classifier_bert.run_bert(strategy, input_meta_data, albert_config,
                               train_input_fn, eval_input_fn)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
