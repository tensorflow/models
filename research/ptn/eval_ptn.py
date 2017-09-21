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

"""Contains evaluation plan for the Im2vox model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow import app

import model_ptn

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir',
                    '',
                    'Directory path containing the input data (tfrecords).')
flags.DEFINE_string(
    'dataset_name', 'shapenet_chair',
    'Dataset name that is to be used for training and evaluation.')
flags.DEFINE_integer('z_dim', 512, '')
flags.DEFINE_integer('f_dim', 64, '')
flags.DEFINE_integer('fc_dim', 1024, '')
flags.DEFINE_integer('num_views', 24, 'Num of viewpoints in the input data.')
flags.DEFINE_integer('image_size', 64,
                     'Input images dimension (pixels) - width & height.')
flags.DEFINE_integer('vox_size', 32, 'Voxel prediction dimension.')
flags.DEFINE_integer('step_size', 24, '')
flags.DEFINE_integer('batch_size', 1, 'Batch size while training.')
flags.DEFINE_float('focal_length', 0.866, '')
flags.DEFINE_float('focal_range', 1.732, '')
flags.DEFINE_string('encoder_name', 'ptn_encoder',
                    'Name of the encoder network being used.')
flags.DEFINE_string('decoder_name', 'ptn_vox_decoder',
                    'Name of the decoder network being used.')
flags.DEFINE_string('projector_name', 'ptn_projector',
                    'Name of the projector network being used.')
# Save options
flags.DEFINE_string('checkpoint_dir', '/tmp/ptn/eval/',
                    'Directory path for saving trained models and other data.')
flags.DEFINE_string('model_name', 'ptn_proj',
                    'Name of the model used in naming the TF job. Must be different for each run.')
flags.DEFINE_string('eval_set', 'val', 'Data partition to form evaluation on.')
# Optimization
flags.DEFINE_float('proj_weight', 10, 'Weighting factor for projection loss.')
flags.DEFINE_float('volume_weight', 0, 'Weighting factor for volume loss.')
flags.DEFINE_float('viewpoint_weight', 1,
                   'Weighting factor for viewpoint loss.')
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0.001, '')
flags.DEFINE_float('clip_gradient_norm', 0, '')
# Summary
flags.DEFINE_integer('save_summaries_secs', 15, '')
flags.DEFINE_integer('eval_interval_secs', 60 * 5, '')
# Distribution
flags.DEFINE_string('master', '', '')

FLAGS = flags.FLAGS


def main(argv=()):
  del argv  # Unused.
  eval_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name, 'train')
  log_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name,
                         'eval_%s' % FLAGS.eval_set)
  if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  g = tf.Graph()

  with g.as_default():
    eval_params = FLAGS
    eval_params.batch_size = 1
    eval_params.step_size = FLAGS.num_views
    ###########
    ## model ##
    ###########
    model = model_ptn.model_PTN(eval_params)
    ##########
    ## data ##
    ##########
    eval_data = model.get_inputs(
        FLAGS.inp_dir,
        FLAGS.dataset_name,
        eval_params.eval_set,
        eval_params.batch_size,
        eval_params.image_size,
        eval_params.vox_size,
        is_training=False)
    inputs = model.preprocess_with_all_views(eval_data)
    ##############
    ## model_fn ##
    ##############
    model_fn = model.get_model_fn(is_training=False, run_projection=False)
    outputs = model_fn(inputs)
    #############
    ## metrics ##
    #############
    names_to_values, names_to_updates = model.get_metrics(inputs, outputs)
    del names_to_values
    ################
    ## evaluation ##
    ################
    num_batches = eval_data['num_samples']
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=eval_dir,
        logdir=log_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  app.run()
