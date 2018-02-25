# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Eval Cross Convolutional Model."""
import io
import os
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import model as cross_conv_model
import reader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('log_root', '/tmp/moving_obj', 'The root dir of output.')
tf.flags.DEFINE_string('data_filepattern',
                       'est',
                       'training data file pattern.')
tf.flags.DEFINE_integer('batch_size', 1, 'Batch size.')
tf.flags.DEFINE_integer('image_size', 64, 'Image height and width.')
tf.flags.DEFINE_float('norm_scale', 1.0, 'Normalize the original image')
tf.flags.DEFINE_float('scale', 10.0,
                      'Scale the image after norm_scale and move the diff '
                      'to the positive realm.')
tf.flags.DEFINE_integer('sequence_length', 2, 'tf.SequenceExample length.')
tf.flags.DEFINE_integer('eval_batch_count', 100,
                        'Average the result this number of examples.')
tf.flags.DEFINE_bool('l2_loss', True, 'If true, include l2_loss.')
tf.flags.DEFINE_bool('reconstr_loss', False, 'If true, include reconstr_loss.')
tf.flags.DEFINE_bool('kl_loss', True, 'If true, include KL loss.')

slim = tf.contrib.slim


def _Eval():
  params = dict()
  params['batch_size'] = FLAGS.batch_size
  params['seq_len'] = FLAGS.sequence_length
  params['image_size'] = FLAGS.image_size
  params['is_training'] = False
  params['norm_scale'] = FLAGS.norm_scale
  params['scale'] = FLAGS.scale
  params['l2_loss'] = FLAGS.l2_loss
  params['reconstr_loss'] = FLAGS.reconstr_loss
  params['kl_loss'] = FLAGS.kl_loss

  eval_dir = os.path.join(FLAGS.log_root, 'eval')

  images = reader.ReadInput(
      FLAGS.data_filepattern, shuffle=False, params=params)
  images *= params['scale']
  # Increase the value makes training much faster.
  image_diff_list = reader.SequenceToImageAndDiff(images)
  model = cross_conv_model.CrossConvModel(image_diff_list, params)
  model.Build()

  summary_writer = tf.summary.FileWriter(eval_dir)
  saver = tf.train.Saver()
  sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  while True:
    time.sleep(60)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      sys.stderr.write('Cannot restore checkpoint: %s\n' % e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      sys.stderr.write('No model to eval yet at %s\n' % FLAGS.log_root)
      continue
    sys.stderr.write('Loading checkpoint %s\n' %
                     ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)
    # Use the empirical distribution of z from training set.
    if not tf.gfile.Exists(os.path.join(FLAGS.log_root, 'z_mean.npy')):
      sys.stderr.write('No z at %s\n' % FLAGS.log_root)
      continue

    with tf.gfile.Open(os.path.join(FLAGS.log_root, 'z_mean.npy')) as f:
      sample_z_mean = np.load(io.BytesIO(f.read()))
    with tf.gfile.Open(
        os.path.join(FLAGS.log_root, 'z_stddev_log.npy')) as f:
      sample_z_stddev_log = np.load(io.BytesIO(f.read()))

    total_loss = 0.0
    for _ in xrange(FLAGS.eval_batch_count):
      loss_val, total_steps, summaries = sess.run(
          [model.loss, model.global_step, model.summary_op],
          feed_dict={model.z_mean: sample_z_mean,
                     model.z_stddev_log: sample_z_stddev_log})
      total_loss += loss_val

    summary_writer.add_summary(summaries, total_steps)
    sys.stderr.write('steps: %d, loss: %f\n' %
                     (total_steps, total_loss / FLAGS.eval_batch_count))


def main(_):
  _Eval()


if __name__ == '__main__':
  tf.app.run()
