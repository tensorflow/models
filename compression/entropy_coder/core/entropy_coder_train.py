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

"""Train an entropy coder model."""

import time

import tensorflow as tf

import code_loader
import config_helper

# pylint: disable=unused-import
from entropy_coder.all_models import all_models
# pylint: enable=unused-import
from entropy_coder.model import model_factory


FLAGS = tf.app.flags.FLAGS

# Hardware resources configuration.
tf.app.flags.DEFINE_string('master', '',
                           """Name of the TensorFlow master to use.""")
tf.app.flags.DEFINE_string('train_dir', None,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('task', None,
                            """Task id of the replica running the training.""")
tf.app.flags.DEFINE_integer('ps_tasks', 0, """Number of tasks in the ps job.
                            If 0 no ps job is used.""")

# Model selection and configuration.
tf.app.flags.DEFINE_string('model', None, """Underlying encoder model.""")
tf.app.flags.DEFINE_string('model_config', None,
                           """Model config protobuf given as text file.""")

# Training data and parameters configuration.
tf.app.flags.DEFINE_string('input_config', None,
                           """Path to the training input config file.""")
tf.app.flags.DEFINE_string('train_config', None,
                           """Path to the training experiment config file.""")


def train():
  if FLAGS.train_dir is None:
    raise ValueError('Parameter train_dir must be provided')
  if FLAGS.task is None:
    raise ValueError('Parameter task must be provided')
  if FLAGS.model is None:
    raise ValueError('Parameter model must be provided')

  input_config_string = config_helper.GetConfigString(FLAGS.input_config)
  input_config = config_helper.InputConfig(input_config_string)

  # Training parameters.
  train_config_string = config_helper.GetConfigString(FLAGS.train_config)
  train_config = config_helper.TrainConfig(train_config_string)

  batch_size = train_config.batch_size
  initial_learning_rate = train_config.learning_rate
  decay_rate = train_config.decay_rate
  samples_per_decay = train_config.samples_per_decay

  # Parameters for learning-rate decay.
  # The formula is decay_rate ** floor(steps / decay_steps).
  decay_steps = samples_per_decay / batch_size
  decay_steps = max(decay_steps, 1)

  first_code = code_loader.ReadFirstCode(input_config.data)
  first_code_height = (
      first_code.features.feature['code_shape'].int64_list.value[0])
  first_code_width = (
      first_code.features.feature['code_shape'].int64_list.value[1])
  max_bit_depth = (
      first_code.features.feature['code_shape'].int64_list.value[2])
  print('Maximum code depth: {}'.format(max_bit_depth))

  with tf.Graph().as_default():
    ps_ops = ["Variable", "VariableV2", "AutoReloadVariable", "VarHandleOp"]
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks,
                                                  ps_ops=ps_ops)):
      codes = code_loader.LoadBinaryCode(
          input_config=input_config,
          batch_size=batch_size)
      if input_config.unique_code_size:
        print('Input code size: {} x {}'.format(first_code_height,
                                                first_code_width))
        codes.set_shape(
            [batch_size, first_code_height, first_code_width, max_bit_depth])
      else:
        codes.set_shape([batch_size, None, None, max_bit_depth])
      codes_effective_shape = tf.shape(codes)

      global_step = tf.contrib.framework.create_global_step()

      # Apply learning-rate decay.
      learning_rate = tf.train.exponential_decay(
          learning_rate=initial_learning_rate,
          global_step=global_step,
          decay_steps=decay_steps,
          decay_rate=decay_rate,
          staircase=True)
      tf.summary.scalar('Learning Rate', learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         epsilon=1.0)

      # Create the entropy coder model.
      model = model_factory.GetModelRegistry().CreateModel(FLAGS.model)
      model_config_string = config_helper.GetConfigString(FLAGS.model_config)
      model.Initialize(global_step, optimizer, model_config_string)
      model.BuildGraph(codes)

      summary_op = tf.summary.merge_all()

      # Verify that the model can actually be trained.
      if model.train_op is None:
        raise ValueError('Input model {} is not trainable'.format(FLAGS.model))

      # We disable the summary thread run by Supervisor class by passing
      # summary_op=None. We still pass save_summaries_secs because it is used by
      # the global step counter thread.
      is_chief = (FLAGS.task == 0)
      sv = tf.train.Supervisor(logdir=FLAGS.train_dir,
                               is_chief=is_chief,
                               global_step=global_step,
                               # saver=model.saver,
                               summary_op=None,
                               save_summaries_secs=120,
                               save_model_secs=600,
                               recovery_wait_secs=30)

      sess = sv.PrepareSession(FLAGS.master)
      sv.StartQueueRunners(sess)

      step = sess.run(global_step)
      print('Trainer initial step: {}.'.format(step))

      # Once everything has been setup properly, save the configs.
      if is_chief:
        config_helper.SaveConfig(FLAGS.train_dir, 'input_config.json',
                                 input_config_string)
        config_helper.SaveConfig(FLAGS.train_dir, 'model_config.json',
                                 model_config_string)
        config_helper.SaveConfig(FLAGS.train_dir, 'train_config.json',
                                 train_config_string)

      # Train the model.
      next_summary_time = time.time()
      while not sv.ShouldStop():
        feed_dict = None

        # Once in a while, update the summaries on the chief worker.
        if is_chief and next_summary_time < time.time():
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          sv.SummaryComputed(sess, summary_str)
          next_summary_time = time.time() + sv.save_summaries_secs
        else:
          tf_tensors = {
              'train': model.train_op,
              'code_length': model.average_code_length
          }
          np_tensors = sess.run(tf_tensors, feed_dict=feed_dict)
          print np_tensors['code_length']

      sv.Stop()


def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
