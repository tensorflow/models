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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in TF 2.x."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.bert import configs as bert_configs
from official.nlp.bert import run_squad_helper
from official.nlp.bert import tokenization
from official.nlp.data import squad_lib as squad_lib_wp
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils


flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')

# More flags can be found in run_squad_helper.
run_squad_helper.define_common_squad_flags()

FLAGS = flags.FLAGS


def train_squad(strategy,
                input_meta_data,
                custom_callbacks=None,
                run_eagerly=False):
  """Run bert squad training."""
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  run_squad_helper.train_squad(strategy, input_meta_data, bert_config,
                               custom_callbacks, run_eagerly)


def predict_squad(strategy, input_meta_data):
  """Makes predictions for a squad dataset."""
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  run_squad_helper.predict_squad(strategy, input_meta_data, tokenizer,
                                 bert_config, squad_lib_wp)


def export_squad(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  run_squad_helper.export_squad(model_export_path, input_meta_data, bert_config)


def main(_):
  # Users should always run this script under TF 2.x

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if FLAGS.mode == 'export_only':
    export_squad(FLAGS.model_export_path, input_meta_data)
    return

  # Configures cluster spec for multi-worker distribution strategy.
  if FLAGS.num_gpus > 0:
    _ = distribution_utils.configure_cluster(FLAGS.worker_hosts,
                                             FLAGS.task_index)
  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      all_reduce_alg=FLAGS.all_reduce_alg,
      tpu_address=FLAGS.tpu)
  if FLAGS.mode in ('train', 'train_and_predict'):
    if FLAGS.log_steps:
      custom_callbacks = [keras_utils.TimeHistory(
          batch_size=FLAGS.train_batch_size,
          log_steps=FLAGS.log_steps,
          logdir=FLAGS.model_dir,
      )]
    else:
      custom_callbacks = None

    train_squad(
        strategy,
        input_meta_data,
        custom_callbacks=custom_callbacks,
        run_eagerly=FLAGS.run_eagerly,
    )
  if FLAGS.mode in ('predict', 'train_and_predict'):
    predict_squad(strategy, input_meta_data)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
