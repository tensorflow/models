# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('pretrained_dir', '/tmp/cifar10_train',
                           """Pretrained model path """)
tf.app.flags.DEFINE_integer('num_epochs', 500,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('training_mini_batch', 128,
                            """Mini batch for training.""")
tf.app.flags.DEFINE_integer('eval_mini_batch', 128,
                            """Mini batch for eval.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")

def train():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.0
    sess = tf.Session(config=config)
    filenames = ['/tmp/cifar10_data/cifar-10-batches-bin/data_batch_%d.bin' % i
                 for i in range(1, 6)]
    global_step = tf.train.get_or_create_global_step()
    batched_images, batched_labels = cifar10_input.read_cifar10(filenames, FLAGS.training_mini_batch, True, True)
    batch_size = tf.shape(batched_images)[0]

    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(batched_images)



    # Calculate loss.
    loss = cifar10.loss(logits, batched_labels)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print(ckpt.model_checkpoint_path)
        restorer = tf.train.Saver([v for v in tf.global_variables()])
        for v in tf.global_variables():
            print(v)
        restorer.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrained_dir))
        # print("variables_to_restore")



    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""
        def begin(self):
            self._step = -1
            self._start_time = time.time()
        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss, batch_size, global_step])  # Asks for loss value.
        def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                duration = current_time - self._start_time
                self._start_time = current_time
                [loss_value, batch_size, global_step] = run_values.results
                examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                sec_per_batch = float(duration / FLAGS.log_frequency)
                format_str = ('%s: step %d, loss = %.2f batch_size = %d global_step = %d (%.1f examples/sec; %.3f '
                            'sec/batch)')
                print (format_str % (datetime.now(), self._step, loss_value,  batch_size, global_step,
                                    examples_per_sec, sec_per_batch))

                      
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.NanTensorHook(loss),
                _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement),
        save_checkpoint_secs=30) as mon_sess:
        while not mon_sess.should_stop():
                mon_sess.run(train_op)

    sess.close()

              


def eval():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    filenames = '/tmp/cifar10_data/cifar-10-batches-bin/test_batch.bin'
    batched_images, batched_labels = cifar10_input.read_cifar10(filenames, FLAGS.eval_mini_batch, False, False)
    batched_logits = cifar10.inference(batched_images)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        restorer = tf.train.Saver([v for v in tf.global_variables()])
        restorer.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(batched_logits, batched_labels, 1)
    total_correct = 0
    total_examples = 0
    while True:
        try:
            value = sess.run(top_k_op)
            total_correct += value.sum()
            total_examples += value.size
        except tf.errors.OutOfRangeError:
            break
    sess.close()
    print("Accuracy is ",total_correct/total_examples)
        


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  #if tf.gfile.Exists(FLAGS.train_dir):
   # tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  for epochs in range(0,FLAGS.num_epochs):
      train()
      eval()


if __name__ == '__main__':
  tf.app.run()
