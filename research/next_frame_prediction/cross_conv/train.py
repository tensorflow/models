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

"""Train the cross convolutional model."""
import os
import sys

import numpy as np
import tensorflow as tf

import model as cross_conv_model
import reader

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('master', '', 'Session address.')
tf.flags.DEFINE_string('log_root', '/tmp/moving_obj', 'The root dir of output.')
tf.flags.DEFINE_string('data_filepattern', '',
                       'training data file pattern.')
tf.flags.DEFINE_integer('image_size', 64, 'Image height and width.')
tf.flags.DEFINE_integer('batch_size', 1, 'Batch size.')
tf.flags.DEFINE_float('norm_scale', 1.0, 'Normalize the original image')
tf.flags.DEFINE_float('scale', 10.0,
                      'Scale the image after norm_scale and move the diff '
                      'to the positive realm.')
tf.flags.DEFINE_integer('sequence_length', 2, 'tf.SequenceExample length.')
tf.flags.DEFINE_float('learning_rate', 0.8, 'Learning rate.')
tf.flags.DEFINE_bool('l2_loss', True, 'If true, include l2_loss.')
tf.flags.DEFINE_bool('reconstr_loss', False, 'If true, include reconstr_loss.')
tf.flags.DEFINE_bool('kl_loss', True, 'If true, include KL loss.')

slim = tf.contrib.slim


def _Train():
  params = dict()
  params['batch_size'] = FLAGS.batch_size
  params['seq_len'] = FLAGS.sequence_length
  params['image_size'] = FLAGS.image_size
  params['is_training'] = True
  params['norm_scale'] = FLAGS.norm_scale
  params['scale'] = FLAGS.scale
  params['learning_rate'] = FLAGS.learning_rate
  params['l2_loss'] = FLAGS.l2_loss
  params['reconstr_loss'] = FLAGS.reconstr_loss
  params['kl_loss'] = FLAGS.kl_loss

  train_dir = os.path.join(FLAGS.log_root, 'train')

  images = reader.ReadInput(FLAGS.data_filepattern, shuffle=True, params=params)
  images *= params['scale']
  # Increase the value makes training much faster.
  image_diff_list = reader.SequenceToImageAndDiff(images)
  model = cross_conv_model.CrossConvModel(image_diff_list, params)
  model.Build()
  tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph())

  summary_writer = tf.summary.FileWriter(train_dir)
  sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                           summary_op=None,
                           is_chief=True,
                           save_model_secs=60,
                           global_step=model.global_step)
  sess = sv.prepare_or_wait_for_session(
      FLAGS.master, config=tf.ConfigProto(allow_soft_placement=True))

  total_loss = 0.0
  step = 0
  sample_z_mean = np.zeros(model.z_mean.get_shape().as_list())
  sample_z_stddev_log = np.zeros(model.z_stddev_log.get_shape().as_list())
  sample_step = 0

  while True:
    _, loss_val, total_steps, summaries, z_mean, z_stddev_log = sess.run(
        [model.train_op, model.loss, model.global_step,
         model.summary_op,
         model.z_mean, model.z_stddev_log])

    sample_z_mean += z_mean
    sample_z_stddev_log += z_stddev_log
    total_loss += loss_val
    step += 1
    sample_step += 1

    if step % 100 == 0:
      summary_writer.add_summary(summaries, total_steps)
      sys.stderr.write('step: %d, loss: %f\n' %
                       (total_steps, total_loss / step))
      total_loss = 0.0
      step = 0

    # Sampled z is used for eval.
    # It seems 10k is better than 1k. Maybe try 100k next?
    if sample_step % 10000 == 0:
      with tf.gfile.Open(os.path.join(FLAGS.log_root, 'z_mean.npy'), 'w') as f:
        np.save(f, sample_z_mean / sample_step)
      with tf.gfile.Open(
          os.path.join(FLAGS.log_root, 'z_stddev_log.npy'), 'w') as f:
        np.save(f, sample_z_stddev_log / sample_step)
      sample_z_mean = np.zeros(model.z_mean.get_shape().as_list())
      sample_z_stddev_log = np.zeros(
          model.z_stddev_log.get_shape().as_list())
      sample_step = 0


def main(_):
  _Train()


if __name__ == '__main__':
  tf.app.run()
