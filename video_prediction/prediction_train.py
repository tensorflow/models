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

"""Code for training the prediction model."""

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from prediction_input import build_tfrecord_input
from prediction_model import construct_model

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

# tf record data location:
DATA_DIR = 'push/push_train'

# local output directory
OUT_DIR = '/tmp/data'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA',
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')


## Helper functions
def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model(object):

  def __init__(self,
               images=None,
               actions=None,
               states=None,
               sequence_length=None,
               reuse_scope=None,
               prefix=None):

    if sequence_length is None:
      sequence_length = FLAGS.sequence_length

    if prefix is None:
        prefix = tf.placeholder(tf.string, [])
    self.prefix = prefix
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []

    # Split into timesteps.
    actions = tf.split(axis=1, num_or_size_splits=int(actions.get_shape()[1]), value=actions)
    actions = [tf.squeeze(act) for act in actions]
    states = tf.split(axis=1, num_or_size_splits=int(states.get_shape()[1]), value=states)
    states = [tf.squeeze(st) for st in states]
    images = tf.split(axis=1, num_or_size_splits=int(images.get_shape()[1]), value=images)
    images = [tf.squeeze(img) for img in images]

    if reuse_scope is None:
      gen_images, gen_states = construct_model(
          images,
          actions,
          states,
          iter_num=self.iter_num,
          k=FLAGS.schedsamp_k,
          use_state=FLAGS.use_state,
          num_masks=FLAGS.num_masks,
          cdna=FLAGS.model == 'CDNA',
          dna=FLAGS.model == 'DNA',
          stp=FLAGS.model == 'STP',
          context_frames=FLAGS.context_frames)
    else:  # If it's a validation or test model.
      with tf.variable_scope(reuse_scope, reuse=True):
        gen_images, gen_states = construct_model(
            images,
            actions,
            states,
            iter_num=self.iter_num,
            k=FLAGS.schedsamp_k,
            use_state=FLAGS.use_state,
            num_masks=FLAGS.num_masks,
            cdna=FLAGS.model == 'CDNA',
            dna=FLAGS.model == 'DNA',
            stp=FLAGS.model == 'STP',
            context_frames=FLAGS.context_frames)

    # L2 loss, PSNR for eval.
    loss, psnr_all = 0.0, 0.0
    for i, x, gx in zip(
        range(len(gen_images)), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      recon_cost = mean_squared_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i
      summaries.append(
          tf.summary.scalar(prefix + '_recon_cost' + str(i), recon_cost))
      summaries.append(tf.summary.scalar(prefix + '_psnr' + str(i), psnr_i))
      loss += recon_cost

    for i, state, gen_state in zip(
        range(len(gen_states)), states[FLAGS.context_frames:],
        gen_states[FLAGS.context_frames - 1:]):
      state_cost = mean_squared_error(state, gen_state) * 1e-4
      summaries.append(
          tf.summary.scalar(prefix + '_state_cost' + str(i), state_cost))
      loss += state_cost
    summaries.append(tf.summary.scalar(prefix + '_psnr_all', psnr_all))
    self.psnr_all = psnr_all

    self.loss = loss = loss / np.float32(len(images) - FLAGS.context_frames)

    summaries.append(tf.summary.scalar(prefix + '_loss', loss))

    self.lr = tf.placeholder_with_default(FLAGS.learning_rate, ())

    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    self.summ_op = tf.summary.merge(summaries)


def main(unused_argv):

  print('Constructing models and inputs.')
  with tf.variable_scope('model', reuse=None) as training_scope:
    images, actions, states = build_tfrecord_input(training=True)
    model = Model(images, actions, states, FLAGS.sequence_length,
                  prefix='train')

  with tf.variable_scope('val_model', reuse=None):
    val_images, val_actions, val_states = build_tfrecord_input(training=False)
    val_model = Model(val_images, val_actions, val_states,
                      FLAGS.sequence_length, training_scope, prefix='val')

  print('Constructing saver.')
  # Make saver.
  saver = tf.train.Saver(
      tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=0)

  # Make training session.
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  summary_writer = tf.summary.FileWriter(
      FLAGS.event_log_dir, graph=sess.graph, flush_secs=10)

  if FLAGS.pretrained_model:
    saver.restore(sess, FLAGS.pretrained_model)

  tf.train.start_queue_runners(sess)

  tf.logging.info('iteration number, cost')

  # Run training.
  for itr in range(FLAGS.num_iterations):
    # Generate new batch of data.
    feed_dict = {model.iter_num: np.float32(itr),
                 model.lr: FLAGS.learning_rate}
    cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op],
                                    feed_dict)

    # Print info: iteration #, cost.
    tf.logging.info(str(itr) + ' ' + str(cost))

    if (itr) % VAL_INTERVAL == 2:
      # Run through validation set.
      feed_dict = {val_model.lr: 0.0,
                   val_model.iter_num: np.float32(itr)}
      _, val_summary_str = sess.run([val_model.train_op, val_model.summ_op],
                                     feed_dict)
      summary_writer.add_summary(val_summary_str, itr)

    if (itr) % SAVE_INTERVAL == 2:
      tf.logging.info('Saving model.')
      saver.save(sess, FLAGS.output_dir + '/model' + str(itr))

    if (itr) % SUMMARY_INTERVAL:
      summary_writer.add_summary(summary_str, itr)

  tf.logging.info('Saving model.')
  saver.save(sess, FLAGS.output_dir + '/model')
  tf.logging.info('Training complete')
  tf.logging.flush()


if __name__ == '__main__':
  app.run()
