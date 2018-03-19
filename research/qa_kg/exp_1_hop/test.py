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

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
import numpy as np
import tensorflow as tf
from config import get_config
from model_n2nmn.assembler import Assembler
from model_n2nmn.model import Model
from util.data_reader import DataReader
from util.data_reader import SampleBuilder
from util.misc import prepare_dirs_and_logger

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('snapshot_name', '00001000', 'snapshot file name')


def main(_):
  config = prepare_dirs_and_logger(config_raw)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)
  config.rng = rng

  config.module_names = ['_key_find', '_key_filter', '_val_desc', '<eos>']
  config.gt_layout_tokens = ['_key_find', '_key_filter', '_val_desc', '<eos>']
  assembler = Assembler(config)

  sample_builder = SampleBuilder(config)
  config = sample_builder.config  # update T_encoder according to data
  data_test = sample_builder.data_all['test']
  data_reader_test = DataReader(
      config, data_test, assembler, shuffle=False, one_pass=True)

  num_vocab_txt = len(sample_builder.dict_all)
  num_vocab_nmn = len(assembler.module_names)
  num_choices = len(sample_builder.dict_all)

  # Network inputs
  text_seq_batch = tf.placeholder(tf.int32, [None, None])
  seq_len_batch = tf.placeholder(tf.int32, [None])

  # The model
  model = Model(
      config,
      sample_builder.kb,
      text_seq_batch,
      seq_len_batch,
      num_vocab_txt=num_vocab_txt,
      num_vocab_nmn=num_vocab_nmn,
      EOS_idx=assembler.EOS_idx,
      num_choices=num_choices,
      decoder_sampling=False)
  compiler = model.compiler
  scores = model.scores

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  snapshot_file = os.path.join(config.model_dir, FLAGS.snapshot_name)
  tf.logging.info('Snapshot file: %s' % snapshot_file)

  snapshot_saver = tf.train.Saver()
  snapshot_saver.restore(sess, snapshot_file)

  # Evaluation metrics
  num_questions = len(data_test.Y)
  tf.logging.info('# of test questions: %d' % num_questions)

  answer_correct = 0
  layout_correct = 0
  layout_valid = 0
  for batch in data_reader_test.batches():
    # set up input and output tensors
    h = sess.partial_run_setup(
        fetches=[model.predicted_tokens, scores],
        feeds=[text_seq_batch, seq_len_batch, compiler.loom_input_tensor])

    # Part 1: Generate module layout
    tokens = sess.partial_run(
        h,
        fetches=model.predicted_tokens,
        feed_dict={
            text_seq_batch: batch['input_seq_batch'],
            seq_len_batch: batch['seq_len_batch']
        })

    # Compute accuracy of the predicted layout
    gt_tokens = batch['gt_layout_batch']
    layout_correct += np.sum(
        np.all(
            np.logical_or(tokens == gt_tokens, gt_tokens == assembler.EOS_idx),
            axis=0))

    # Assemble the layout tokens into network structure
    expr_list, expr_validity_array = assembler.assemble(tokens)
    layout_valid += np.sum(expr_validity_array)
    labels = batch['ans_label_batch']
    # Build TensorFlow Fold input for NMN
    expr_feed = compiler.build_feed_dict(expr_list)

    # Part 2: Run NMN and learning steps
    scores_val = sess.partial_run(h, scores, feed_dict=expr_feed)

    # Compute accuracy
    predictions = np.argmax(scores_val, axis=1)
    answer_correct += np.sum(
        np.logical_and(expr_validity_array, predictions == labels))

  answer_accuracy = answer_correct * 1.0 / num_questions
  layout_accuracy = layout_correct * 1.0 / num_questions
  layout_validity = layout_valid * 1.0 / num_questions

  tf.logging.info('test answer accuracy = %f, '
                  'test layout accuracy = %f, '
                  'test layout validity = %f' %
                  (answer_accuracy, layout_accuracy, layout_validity))


if __name__ == '__main__':
  config_raw, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
