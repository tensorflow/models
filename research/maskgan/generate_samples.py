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

"""Generate samples from the MaskGAN.

Launch command:
  python generate_samples.py
  --data_dir=/tmp/data/imdb  --data_set=imdb
  --batch_size=256 --sequence_length=20 --base_directory=/tmp/imdb
  --hparams="gen_rnn_size=650,dis_rnn_size=650,gen_num_layers=2,
  gen_vd_keep_prob=1.0" --generator_model=seq2seq_vd
  --discriminator_model=seq2seq_vd --is_present_rate=0.5
  --maskgan_ckpt=/tmp/model.ckpt-45494
  --seq2seq_share_embedding=True --dis_share_embedding=True
  --attention_option=luong --mask_strategy=contiguous --baseline_method=critic
  --number_epochs=4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import os
# Dependency imports

import numpy as np
from six.moves import xrange
import tensorflow as tf

import train_mask_gan
from data import imdb_loader
from data import ptb_loader

# Data.
from model_utils import helper
from model_utils import model_utils

SAMPLE_TRAIN = 'TRAIN'
SAMPLE_VALIDATION = 'VALIDATION'

## Sample Generation.
## Binary and setup FLAGS.
tf.app.flags.DEFINE_enum('sample_mode', 'TRAIN',
                         [SAMPLE_TRAIN, SAMPLE_VALIDATION],
                         'Dataset to sample from.')
tf.app.flags.DEFINE_string('output_path', '/tmp', 'Model output directory.')
tf.app.flags.DEFINE_boolean(
    'output_masked_logs', False,
    'Whether to display for human evaluation (show masking).')
tf.app.flags.DEFINE_integer('number_epochs', 1,
                            'The number of epochs to produce.')

FLAGS = tf.app.flags.FLAGS


def get_iterator(data):
  """Return the data iterator."""
  if FLAGS.data_set == 'ptb':
    iterator = ptb_loader.ptb_iterator(data, FLAGS.batch_size,
                                       FLAGS.sequence_length,
                                       FLAGS.epoch_size_override)
  elif FLAGS.data_set == 'imdb':
    iterator = imdb_loader.imdb_iterator(data, FLAGS.batch_size,
                                         FLAGS.sequence_length)
  return iterator


def convert_to_human_readable(id_to_word, arr, p, max_num_to_print):
  """Convert a np.array of indices into words using id_to_word dictionary.
  Return max_num_to_print results.
  """

  assert arr.ndim == 2

  samples = []
  for sequence_id in xrange(min(len(arr), max_num_to_print)):
    sample = []
    for i, index in enumerate(arr[sequence_id, :]):
      if p[sequence_id, i] == 1:
        sample.append(str(id_to_word[index]))
      else:
        sample.append('*' + str(id_to_word[index]))
    buffer_str = ' '.join(sample)
    samples.append(buffer_str)
  return samples


def write_unmasked_log(log, id_to_word, sequence_eval):
  """Helper function for logging evaluated sequences without mask."""
  indices_arr = np.asarray(sequence_eval)
  samples = helper.convert_to_human_readable(id_to_word, indices_arr,
                                             FLAGS.batch_size)
  for sample in samples:
    log.write(sample + '\n')
  log.flush()
  return samples


def write_masked_log(log, id_to_word, sequence_eval, present_eval):
  indices_arr = np.asarray(sequence_eval)
  samples = convert_to_human_readable(id_to_word, indices_arr, present_eval,
                                      FLAGS.batch_size)
  for sample in samples:
    log.write(sample + '\n')
  log.flush()
  return samples


def generate_logs(sess, model, log, id_to_word, feed):
  """Impute Sequences using the model for a particular feed and send it to
  logs.
  """
  # Impute Sequences.
  [p, inputs_eval, sequence_eval] = sess.run(
      [model.present, model.inputs, model.fake_sequence], feed_dict=feed)

  # Add the 0th time-step for coherence.
  first_token = np.expand_dims(inputs_eval[:, 0], axis=1)
  sequence_eval = np.concatenate((first_token, sequence_eval), axis=1)

  # 0th token always present.
  p = np.concatenate((np.ones((FLAGS.batch_size, 1)), p), axis=1)

  if FLAGS.output_masked_logs:
    samples = write_masked_log(log, id_to_word, sequence_eval, p)
  else:
    samples = write_unmasked_log(log, id_to_word, sequence_eval)
  return samples


def generate_samples(hparams, data, id_to_word, log_dir, output_file):
  """"Generate samples.

    Args:
      hparams:  Hyperparameters for the MaskGAN.
      data: Data to evaluate.
      id_to_word: Dictionary of indices to words.
      log_dir: Log directory.
      output_file:  Output file for the samples.
  """
  # Boolean indicating operational mode.
  is_training = False

  # Set a random seed to keep fixed mask.
  np.random.seed(0)

  with tf.Graph().as_default():
    # Construct the model.
    model = train_mask_gan.create_MaskGAN(hparams, is_training)

    ## Retrieve the initial savers.
    init_savers = model_utils.retrieve_init_savers(hparams)

    ## Initial saver function to supervisor.
    init_fn = partial(model_utils.init_fn, init_savers)

    is_chief = FLAGS.task == 0

    # Create the supervisor.  It will take care of initialization, summaries,
    # checkpoints, and recovery.
    sv = tf.Supervisor(
        logdir=log_dir,
        is_chief=is_chief,
        saver=model.saver,
        global_step=model.global_step,
        recovery_wait_secs=30,
        summary_op=None,
        init_fn=init_fn)

    # Get an initialized, and possibly recovered session.  Launch the
    # services: Checkpointing, Summaries, step counting.
    #
    # When multiple replicas of this program are running the services are
    # only launched by the 'chief' replica.
    with sv.managed_session(
        FLAGS.master, start_standard_services=False) as sess:

      # Generator statefulness over the epoch.
      [gen_initial_state_eval, fake_gen_initial_state_eval] = sess.run(
          [model.eval_initial_state, model.fake_gen_initial_state])

      for n in xrange(FLAGS.number_epochs):
        print('Epoch number: %d' % n)
        # print('Percent done: %.2f' % float(n) / float(FLAGS.number_epochs))
        iterator = get_iterator(data)
        for x, y, _ in iterator:
          if FLAGS.eval_language_model:
            is_present_rate = 0.
          else:
            is_present_rate = FLAGS.is_present_rate
          tf.logging.info(
              'Evaluating on is_present_rate=%.3f.' % is_present_rate)

          model_utils.assign_percent_real(sess, model.percent_real_update,
                                          model.new_rate, is_present_rate)

          # Randomly mask out tokens.
          p = model_utils.generate_mask()

          eval_feed = {model.inputs: x, model.targets: y, model.present: p}

          if FLAGS.data_set == 'ptb':
            # Statefulness for *evaluation* Generator.
            for i, (c, h) in enumerate(model.eval_initial_state):
              eval_feed[c] = gen_initial_state_eval[i].c
              eval_feed[h] = gen_initial_state_eval[i].h

            # Statefulness for the Generator.
            for i, (c, h) in enumerate(model.fake_gen_initial_state):
              eval_feed[c] = fake_gen_initial_state_eval[i].c
              eval_feed[h] = fake_gen_initial_state_eval[i].h

          [gen_initial_state_eval, fake_gen_initial_state_eval, _] = sess.run(
              [
                  model.eval_final_state, model.fake_gen_final_state,
                  model.global_step
              ],
              feed_dict=eval_feed)

          generate_logs(sess, model, output_file, id_to_word, eval_feed)
      output_file.close()
      print('Closing output_file.')
      return


def main(_):
  hparams = train_mask_gan.create_hparams()
  log_dir = FLAGS.base_directory

  tf.gfile.MakeDirs(FLAGS.output_path)
  output_file = tf.gfile.GFile(
      os.path.join(FLAGS.output_path, 'reviews.txt'), mode='w')

  # Load data set.
  if FLAGS.data_set == 'ptb':
    raw_data = ptb_loader.ptb_raw_data(FLAGS.data_dir)
    train_data, valid_data, _, _ = raw_data
  elif FLAGS.data_set == 'imdb':
    raw_data = imdb_loader.imdb_raw_data(FLAGS.data_dir)
    train_data, valid_data = raw_data
  else:
    raise NotImplementedError

  # Generating more data on train set.
  if FLAGS.sample_mode == SAMPLE_TRAIN:
    data_set = train_data
  elif FLAGS.sample_mode == SAMPLE_VALIDATION:
    data_set = valid_data
  else:
    raise NotImplementedError

  # Dictionary and reverse dictionry.
  if FLAGS.data_set == 'ptb':
    word_to_id = ptb_loader.build_vocab(
        os.path.join(FLAGS.data_dir, 'ptb.train.txt'))
  elif FLAGS.data_set == 'imdb':
    word_to_id = imdb_loader.build_vocab(
        os.path.join(FLAGS.data_dir, 'vocab.txt'))
  id_to_word = {v: k for k, v in word_to_id.iteritems()}

  FLAGS.vocab_size = len(id_to_word)
  print('Vocab size: %d' % FLAGS.vocab_size)

  generate_samples(hparams, data_set, id_to_word, log_dir, output_file)


if __name__ == '__main__':
  tf.app.run()
