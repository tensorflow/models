# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet Train/Eval module.
"""
import sys
import time

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '', 'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '', 'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')


def train(hps):
  """Training loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)

  sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                           is_chief=True,
                           summary_op=None,
                           save_summaries_secs=60,
                           save_model_secs=300,
                           global_step=model.global_step)
  sess = sv.prepare_or_wait_for_session(
      config=tf.ConfigProto(allow_soft_placement=True))

  step = 0
  lrn_rate = 0.1

  while not sv.should_stop():
    (_, summaries, loss, predictions, truth, train_step) = sess.run(
        [model.train_op, model.summaries, model.cost, model.predictions,
         model.labels, model.global_step],
        feed_dict={model.lrn_rate: lrn_rate})

    if train_step < 40000:
      lrn_rate = 0.1
    elif train_step < 60000:
      lrn_rate = 0.01
    elif train_step < 80000:
      lrn_rate = 0.001
    else:
      lrn_rate = 0.0001

    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    precision = np.mean(truth == predictions)

    step += 1
    if step % 100 == 0:
      precision_summ = tf.Summary()
      precision_summ.value.add(
          tag='Precision', simple_value=precision)
      summary_writer.add_summary(precision_summ, train_step)
      summary_writer.add_summary(summaries, train_step)
      tf.logging.info('loss: %.3f, precision: %.3f\n' % (loss, precision))
      summary_writer.flush()

  sv.Stop()


def evaluate(hps):
  """Eval loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    time.sleep(60)
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    for _ in xrange(FLAGS.eval_batch_count):
      (summaries, loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f\n' %
                    (loss, precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 100

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.app.run()
