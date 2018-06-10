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

"""Train the model."""

# Example usage:
#
# python train.py \
#   --logtostderr \
#   --data_dir ~/vid2depth/data/kitti_raw_eigen \
#   --seq_length 3 \
#   --reconstr_weight 0.85 \
#   --smooth_weight 0.05 \
#   --ssim_weight 0.15 \
#   --icp_weight 0.1 \
#   --checkpoint_dir ~/vid2depth/checkpoints

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import model
import numpy as np
import tensorflow as tf
import util

gfile = tf.gfile

HOME_DIR = os.path.expanduser('~')
DEFAULT_DATA_DIR = os.path.join(HOME_DIR, 'vid2depth/data/kitti_raw_eigen')
DEFAULT_CHECKPOINT_DIR = os.path.join(HOME_DIR, 'vid2depth/checkpoints')

flags.DEFINE_string('data_dir', DEFAULT_DATA_DIR, 'Preprocessed data.')
flags.DEFINE_float('learning_rate', 0.0002, 'Adam learning rate.')
flags.DEFINE_float('beta1', 0.9, 'Adam momentum.')
flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')
flags.DEFINE_float('smooth_weight', 0.05, 'Smoothness loss weight.')
flags.DEFINE_float('ssim_weight', 0.15, 'SSIM loss weight.')
flags.DEFINE_float('icp_weight', 0.0, 'ICP loss weight.')
flags.DEFINE_integer('batch_size', 4, 'The size of a sample batch')
flags.DEFINE_integer('img_height', 128, 'Input frame height.')
flags.DEFINE_integer('img_width', 416, 'Input frame width.')
# Note: Training time grows linearly with sequence length.  Use 2 or 3.
flags.DEFINE_integer('seq_length', 3, 'Number of frames in sequence.')
flags.DEFINE_string('pretrained_ckpt', None, 'Path to checkpoint with '
                    'pretrained weights.  Do not include .data* extension.')
flags.DEFINE_string('checkpoint_dir', DEFAULT_CHECKPOINT_DIR,
                    'Directory to save model checkpoints.')
flags.DEFINE_integer('train_steps', 200000, 'Number of training steps.')
flags.DEFINE_integer('summary_freq', 100, 'Save summaries every N steps.')
flags.DEFINE_bool('legacy_mode', False, 'Whether to limit losses to using only '
                  'the middle frame in sequence as the target frame.')
FLAGS = flags.FLAGS

# Maximum number of checkpoints to keep.
MAX_TO_KEEP = 100


def main(_):
  # Fixed seed for repeatability
  seed = 8964
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if FLAGS.legacy_mode and FLAGS.seq_length < 3:
    raise ValueError('Legacy mode supports sequence length > 2 only.')

  if not gfile.Exists(FLAGS.checkpoint_dir):
    gfile.MakeDirs(FLAGS.checkpoint_dir)

  train_model = model.Model(data_dir=FLAGS.data_dir,
                            is_training=True,
                            learning_rate=FLAGS.learning_rate,
                            beta1=FLAGS.beta1,
                            reconstr_weight=FLAGS.reconstr_weight,
                            smooth_weight=FLAGS.smooth_weight,
                            ssim_weight=FLAGS.ssim_weight,
                            icp_weight=FLAGS.icp_weight,
                            batch_size=FLAGS.batch_size,
                            img_height=FLAGS.img_height,
                            img_width=FLAGS.img_width,
                            seq_length=FLAGS.seq_length,
                            legacy_mode=FLAGS.legacy_mode)

  train(train_model, FLAGS.pretrained_ckpt, FLAGS.checkpoint_dir,
        FLAGS.train_steps, FLAGS.summary_freq)


def train(train_model, pretrained_ckpt, checkpoint_dir, train_steps,
          summary_freq):
  """Train model."""
  if pretrained_ckpt is not None:
    vars_to_restore = util.get_vars_to_restore(pretrained_ckpt)
    pretrain_restorer = tf.train.Saver(vars_to_restore)
  vars_to_save = util.get_vars_to_restore()
  saver = tf.train.Saver(vars_to_save + [train_model.global_step],
                         max_to_keep=MAX_TO_KEEP)
  sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                           saver=None)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with sv.managed_session(config=config) as sess:
    if pretrained_ckpt is not None:
      logging.info('Restoring pretrained weights from %s', pretrained_ckpt)
      pretrain_restorer.restore(sess, pretrained_ckpt)
    logging.info('Attempting to resume training from %s...', checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      saver.restore(sess, checkpoint)

    logging.info('Training...')
    start_time = time.time()
    last_summary_time = time.time()
    steps_per_epoch = train_model.reader.steps_per_epoch
    step = 1
    while step <= train_steps:
      fetches = {
          'train': train_model.train_op,
          'global_step': train_model.global_step,
          'incr_global_step': train_model.incr_global_step
      }

      if step % summary_freq == 0:
        fetches['loss'] = train_model.total_loss
        fetches['summary'] = sv.summary_op

      results = sess.run(fetches)
      global_step = results['global_step']

      if step % summary_freq == 0:
        sv.summary_writer.add_summary(results['summary'], global_step)
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        this_cycle = time.time() - last_summary_time
        last_summary_time += this_cycle
        logging.info(
            'Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f',
            train_epoch, train_step, steps_per_epoch, this_cycle,
            time.time() - start_time, results['loss'])

      if step % steps_per_epoch == 0:
        logging.info('[*] Saving checkpoint to %s...', checkpoint_dir)
        saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                   global_step=global_step)

      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1


if __name__ == '__main__':
  app.run(main)
