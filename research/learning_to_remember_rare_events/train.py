# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# ==============================================================================
r"""Script for training model.

Simple command to get up and running:
  python train.py --memory_size=8192 \
      --batch_size=16 --validation_length=50 \
      --episode_width=5 --episode_length=30
"""

import logging
import os
import random

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils
import model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('rep_dim', 128,
                        'dimension of keys to use in memory')
tf.flags.DEFINE_integer('episode_length', 100, 'length of episode')
tf.flags.DEFINE_integer('episode_width', 5,
                        'number of distinct labels in a single episode')
tf.flags.DEFINE_integer('memory_size', None, 'number of slots in memory. '
                        'Leave as None to default to episode length')
tf.flags.DEFINE_integer('batch_size', 16, 'batch size')
tf.flags.DEFINE_integer('num_episodes', 100000, 'number of training episodes')
tf.flags.DEFINE_integer('validation_frequency', 20,
                        'every so many training episodes, '
                        'assess validation accuracy')
tf.flags.DEFINE_integer('validation_length', 10,
                        'number of episodes to use to compute '
                        'validation accuracy')
tf.flags.DEFINE_integer('seed', 888, 'random seed for training sampling')
tf.flags.DEFINE_string('save_dir', '', 'directory to save model to')
tf.flags.DEFINE_bool('use_lsh', False,
                     'use locality-sensitive hashing '
                     '(NOTE: not fully tested)')


class Trainer(object):
  """Class that takes care of training, validating, and checkpointing model."""

  def __init__(self, train_data, valid_data, input_dim, output_dim=None):
    self.train_data = train_data
    self.valid_data = valid_data
    self.input_dim = input_dim

    self.rep_dim = FLAGS.rep_dim
    self.episode_length = FLAGS.episode_length
    self.episode_width = FLAGS.episode_width
    self.batch_size = FLAGS.batch_size
    self.memory_size = (self.episode_length * self.batch_size
                        if FLAGS.memory_size is None else FLAGS.memory_size)
    self.use_lsh = FLAGS.use_lsh

    self.output_dim = (output_dim if output_dim is not None
                       else self.episode_width)

  def get_model(self):
    # vocab size is the number of distinct values that
    # could go into the memory key-value storage
    vocab_size = self.episode_width * self.batch_size
    return model.Model(
        self.input_dim, self.output_dim, self.rep_dim, self.memory_size,
        vocab_size, use_lsh=self.use_lsh)

  def sample_episode_batch(self, data,
                           episode_length, episode_width, batch_size):
    """Generates a random batch for training or validation.

    Structures each element of the batch as an 'episode'.
    Each episode contains episode_length examples and
    episode_width distinct labels.

    Args:
      data: A dictionary mapping label to list of examples.
      episode_length: Number of examples in each episode.
      episode_width: Distinct number of labels in each episode.
      batch_size: Batch size (number of episodes).

    Returns:
      A tuple (x, y) where x is a list of batches of examples
      with size episode_length and y is a list of batches of labels.
    """

    episodes_x = [[] for _ in xrange(episode_length)]
    episodes_y = [[] for _ in xrange(episode_length)]
    assert len(data) >= episode_width
    keys = data.keys()
    for b in xrange(batch_size):
      episode_labels = random.sample(keys, episode_width)
      remainder = episode_length % episode_width
      remainders = [0] * (episode_width - remainder) + [1] * remainder
      episode_x = [
          random.sample(data[lab],
                        r + (episode_length - remainder) // episode_width)
          for lab, r in zip(episode_labels, remainders)]
      episode = sum([[(x, i, ii) for ii, x in enumerate(xx)]
                     for i, xx in enumerate(episode_x)], [])
      random.shuffle(episode)
      # Arrange episode so that each distinct label is seen before moving to
      # 2nd showing
      episode.sort(key=lambda elem: elem[2])
      assert len(episode) == episode_length
      for i in xrange(episode_length):
        episodes_x[i].append(episode[i][0])
        episodes_y[i].append(episode[i][1] + b * episode_width)

    return ([np.array(xx).astype('float32') for xx in episodes_x],
            [np.array(yy).astype('int32') for yy in episodes_y])

  def compute_correct(self, ys, y_preds):
    return np.mean(np.equal(y_preds, np.array(ys)))

  def individual_compute_correct(self, y, y_pred):
    return y_pred == y

  def run(self):
    """Performs training.

    Trains a model using episodic training.
    Every so often, runs some evaluations on validation data.
    """

    train_data, valid_data = self.train_data, self.valid_data
    input_dim, output_dim = self.input_dim, self.output_dim
    rep_dim, episode_length = self.rep_dim, self.episode_length
    episode_width, memory_size = self.episode_width, self.memory_size
    batch_size = self.batch_size

    train_size = len(train_data)
    valid_size = len(valid_data)
    logging.info('train_size (number of labels) %d', train_size)
    logging.info('valid_size (number of labels) %d', valid_size)
    logging.info('input_dim %d', input_dim)
    logging.info('output_dim %d', output_dim)
    logging.info('rep_dim %d', rep_dim)
    logging.info('episode_length %d', episode_length)
    logging.info('episode_width %d', episode_width)
    logging.info('memory_size %d', memory_size)
    logging.info('batch_size %d', batch_size)

    assert all(len(v) >= float(episode_length) / episode_width
               for v in train_data.values())
    assert all(len(v) >= float(episode_length) / episode_width
               for v in valid_data.values())

    output_dim = episode_width
    self.model = self.get_model()
    self.model.setup()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=10)
    ckpt = None
    if FLAGS.save_dir:
      ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
    if ckpt and ckpt.model_checkpoint_path:
      logging.info('restoring from %s', ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)

    logging.info('starting now')
    losses = []
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    for i in xrange(FLAGS.num_episodes):
      x, y = self.sample_episode_batch(
          train_data, episode_length, episode_width, batch_size)
      outputs = self.model.episode_step(sess, x, y, clear_memory=True)
      loss = outputs
      losses.append(loss)

      if i % FLAGS.validation_frequency == 0:
        logging.info('episode batch %d, avg train loss %f',
                     i, np.mean(losses))
        losses = []

        # validation
        correct = []
        num_shots = episode_length // episode_width
        correct_by_shot = dict((k, []) for k in xrange(num_shots))
        for _ in xrange(FLAGS.validation_length):
          x, y = self.sample_episode_batch(
              valid_data, episode_length, episode_width, 1)
          outputs = self.model.episode_predict(
              sess, x, y, clear_memory=True)
          y_preds = outputs
          correct.append(self.compute_correct(np.array(y), y_preds))

          # compute per-shot accuracies
          seen_counts = [0] * episode_width
          # loop over episode steps
          for yy, yy_preds in zip(y, y_preds):
            # loop over batch examples
            yyy, yyy_preds = int(yy[0]), int(yy_preds[0])
            count = seen_counts[yyy % episode_width]
            if count in correct_by_shot:
              correct_by_shot[count].append(
                self.individual_compute_correct(yyy, yyy_preds))
            seen_counts[yyy % episode_width] = count + 1

        logging.info('validation overall accuracy %f', np.mean(correct))
        logging.info('%d-shot: %.3f, ' * num_shots,
                     *sum([[k, np.mean(correct_by_shot[k])]
                           for k in xrange(num_shots)], []))

        if saver and FLAGS.save_dir:
          saved_file = saver.save(sess,
                                  os.path.join(FLAGS.save_dir, 'model.ckpt'),
                                  global_step=self.model.global_step)
          logging.info('saved model to %s', saved_file)


def main(unused_argv):
  train_data, valid_data = data_utils.get_data()
  trainer = Trainer(train_data, valid_data, data_utils.IMAGE_NEW_SIZE ** 2)
  trainer.run()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
