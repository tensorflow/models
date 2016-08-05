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
import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import resnet_model
from datasets import DataSet
import settings


FLAGS = tf.app.flags.FLAGS


def train(hps):
  """Training loop."""
  train_set = DataSet(list_path=FLAGS.train_list_path, 
                      batch_size=FLAGS.batch_size, 
                      height=FLAGS.image_height, 
                      width=FLAGS.image_width, 
                      depth=FLAGS.image_depth,
                      delimeter=FLAGS.delimeter)
  images, labels = train_set.inputs(train=True)

  model = resnet_model.ResNet(hps, images, labels, train=True)
  model.inference()

  # Create a saver.
  saver = tf.train.Saver(tf.all_variables(), max_to_keep=12)

  # Build the summary operation from the last tower summaries.
  # summary_op = tf.merge_summary(summaries)

  # Build an initialization operation to run below.
  init = tf.initialize_all_variables()

  sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True))
  sess.run(init)


  if tf.train.get_checkpoint_state(FLAGS.train_dir):
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
      print ("No checkpoint to continue from in", FLAGS.train_dir)
      sys.exit(1)
    print ("Training continued from ", latest)
    start = int(latest[latest.find('ckpt')+5:])+1
    saver.restore(sess, latest)
  else:
    start = 0
    print ("Training begins... ")

  # Start the queue runners.
  tf.train.start_queue_runners(sess=sess)

  summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, 
                                          graph_def=sess.graph.as_graph_def(add_shapes=True))

  next_checkpoint_time = time.time() + FLAGS.save_interval_secs
  next_summary_time = time.time() + FLAGS.save_summaries_secs

  for step in xrange(start, FLAGS.max_steps):
    start_time = time.time()
    (_, summaries, loss, train_step, learning_rate) = sess.run([model.train_op, 
                                                                 model.summaries, 
                                                                 model.loss, 
                                                                 model.global_step,
                                                                 model.learning_rate])
    duration = time.time() - start_time
    assert not np.isnan(loss), 'Model diverged with loss = NaN'

    if step % 10 == 0:
      examples_per_sec = FLAGS.batch_size / float(duration)
      
      # (predictions, truth) = sess.run([model.predictions, model.one_hot_labels])
      #predictions = np.argmax(predictions, axis=1)
      #truth = np.argmax(truth, axis=1)
      #for (t, p) in zip(truth, predictions):
      #  if t == p:
      #    correct_prediction += 1
      #  total_prediction += 1
      #precision = float(correct_prediction) / total_prediction
      #correct_prediction = total_prediction = 0

      format_str = ('%s: step %d, lr = %.4f, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step, learning_rate, loss,
                          examples_per_sec, duration))



    if next_summary_time < time.time():
      summary_writer.add_summary(summaries, step)
      print('Summary operation done.')
      next_summary_time += FLAGS.save_summaries_secs

    if next_checkpoint_time < time.time():
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=train_step)
      print('Checkpoint saved.')
      next_checkpoint_time += FLAGS.save_interval_secs

       #   step += 1
  #   if step % 10 == 0:
  #     precision_summ = tf.Summary()
  #     precision_summ.value.add(
  #         tag='Precision', simple_value=precision)
  #     summary_writer.add_summary(precision_summ, train_step)
  #     summary_writer.add_summary(summaries, train_step)
  #     tf.logging.info('step: %d, lr: %.5f, loss: %.3f, precision: %.3f\n' % (step, learning_rate, loss, precision))
  #     summary_writer.flush()

  # sv.Stop()


def main(_):
  hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                             num_classes=FLAGS.num_classes,
                             num_gpus=FLAGS.num_gpus,
                             initial_learning_rate=FLAGS.initial_learning_rate,
                             lr_decay_steps=FLAGS.lr_decay_steps,
                             lr_decay_factor=FLAGS.lr_decay_factor,
                             optimizer=FLAGS.optimizer,
                             num_layers=FLAGS.num_layers,
                             prob_depth=0.5,
                             use_bottleneck=True,
                             weight_decay_rate=0.0001,
                             relu_leakiness=0)

  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  #with tf.device(dev):
  #  if FLAGS.mode == 'train':
  #    train(hps)
  #  elif FLAGS.mode == 'eval':
  #    evaluate(hps)
  train(hps) 

if __name__ == '__main__':
  tf.app.run()
