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

"""Contains evaluation plan for the Rotator model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow import app

import model_rotator as model

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir',
                    '',
                    'Directory path containing the input data (tfrecords).')
flags.DEFINE_string(
    'dataset_name', 'shapenet_chair',
    'Dataset name that is to be used for training and evaluation.')
flags.DEFINE_integer('z_dim', 512, '')
flags.DEFINE_integer('a_dim', 3, '')
flags.DEFINE_integer('f_dim', 64, '')
flags.DEFINE_integer('fc_dim', 1024, '')
flags.DEFINE_integer('num_views', 24, 'Num of viewpoints in the input data.')
flags.DEFINE_integer('image_size', 64,
                     'Input images dimension (pixels) - width & height.')
flags.DEFINE_integer('step_size', 24, '')
flags.DEFINE_integer('batch_size', 2, '')
flags.DEFINE_string('encoder_name', 'ptn_encoder',
                    'Name of the encoder network being used.')
flags.DEFINE_string('decoder_name', 'ptn_im_decoder',
                    'Name of the decoder network being used.')
flags.DEFINE_string('rotator_name', 'ptn_rotator',
                    'Name of the rotator network being used.')
# Save options
flags.DEFINE_string('checkpoint_dir', '/tmp/ptn_train/',
                    'Directory path for saving trained models and other data.')
flags.DEFINE_string('model_name', 'ptn_proj',
                    'Name of the model used in naming the TF job. Must be different for each run.')
# Optimization
flags.DEFINE_float('image_weight', 10, '')
flags.DEFINE_float('mask_weight', 1, '')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0.001, '')
flags.DEFINE_float('clip_gradient_norm', 0, '')
# Summary
flags.DEFINE_integer('save_summaries_secs', 15, '')
flags.DEFINE_integer('eval_interval_secs', 60 * 5, '')
# Scheduling
flags.DEFINE_string('master', '', '')

FLAGS = flags.FLAGS


def main(argv=()):
  del argv  # Unused.
  eval_dir = os.path.join(FLAGS.checkpoint_dir,
                          FLAGS.model_name, 'train')
  log_dir = os.path.join(FLAGS.checkpoint_dir,
                         FLAGS.model_name, 'eval')

  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  g = tf.Graph()

  if FLAGS.step_size < FLAGS.num_views:
    raise ValueError('Impossible step_size, must not be less than num_views.')

  g = tf.Graph()
  with g.as_default():
    ##########
    ## data ##
    ##########
    val_data = model.get_inputs(
        FLAGS.inp_dir,
        FLAGS.dataset_name,
        'val',
        FLAGS.batch_size,
        FLAGS.image_size,
        is_training=False)
    inputs = model.preprocess(val_data, FLAGS.step_size)
    ###########
    ## model ##
    ###########
    model_fn = model.get_model_fn(FLAGS, is_training=False)
    outputs = model_fn(inputs)
    #############
    ## metrics ##
    #############
    names_to_values, names_to_updates = model.get_metrics(
        inputs, outputs, FLAGS)
    del names_to_values
    ################
    ## evaluation ##
    ################
    num_batches = int(val_data['num_samples'] / FLAGS.batch_size)
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=eval_dir,
        logdir=log_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  app.run()
