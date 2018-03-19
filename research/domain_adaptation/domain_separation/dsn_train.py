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

"""Training for Domain Separation Networks (DSNs)."""
from __future__ import division

import tensorflow as tf

from domain_adaptation.datasets import dataset_factory
import dsn

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The number of images in each batch.')

tf.app.flags.DEFINE_string('source_dataset', 'pose_synthetic',
                           'Source dataset to train on.')

tf.app.flags.DEFINE_string('target_dataset', 'pose_real',
                           'Target dataset to train on.')

tf.app.flags.DEFINE_string('target_labeled_dataset', 'none',
                           'Target dataset to train on.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('master', '',
                           'BNS name of the TensorFlow master to use.')

tf.app.flags.DEFINE_string('train_log_dir', '/tmp/da/',
                           'Directory where to write event logs.')

tf.app.flags.DEFINE_string(
    'layers_to_regularize', 'fc3',
    'Comma-separated list of layer names to use MMD regularization on.')

tf.app.flags.DEFINE_float('learning_rate', .01, 'The learning rate')

tf.app.flags.DEFINE_float('alpha_weight', 1e-6,
                          'The coefficient for scaling the reconstruction '
                          'loss.')

tf.app.flags.DEFINE_float(
    'beta_weight', 1e-6,
    'The coefficient for scaling the private/shared difference loss.')

tf.app.flags.DEFINE_float(
    'gamma_weight', 1e-6,
    'The coefficient for scaling the shared encoding similarity loss.')

tf.app.flags.DEFINE_float('pose_weight', 0.125,
                          'The coefficient for scaling the pose loss.')

tf.app.flags.DEFINE_float(
    'weight_decay', 1e-6,
    'The coefficient for the L2 regularization applied for all weights.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 60,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None,
    'The maximum number of gradient steps. Use None to train indefinitely.')

tf.app.flags.DEFINE_integer(
    'domain_separation_startpoint', 1,
    'The global step to add the domain separation losses.')

tf.app.flags.DEFINE_integer(
    'bipartite_assignment_top_k', 3,
    'The number of top-k matches to use in bipartite matching adaptation.')

tf.app.flags.DEFINE_float('decay_rate', 0.95, 'Learning rate decay factor.')

tf.app.flags.DEFINE_integer('decay_steps', 20000, 'Learning rate decay steps.')

tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum value.')

tf.app.flags.DEFINE_bool('use_separation', False,
                         'Use our domain separation model.')

tf.app.flags.DEFINE_bool('use_logging', False, 'Debugging messages.')

tf.app.flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

tf.app.flags.DEFINE_string('decoder_name', 'small_decoder',
                           'The decoder to use.')
tf.app.flags.DEFINE_string('encoder_name', 'default_encoder',
                           'The encoder to use.')

################################################################################
# Flags that control the architecture and losses
################################################################################
tf.app.flags.DEFINE_string(
    'similarity_loss', 'grl',
    'The method to use for encouraging the common encoder codes to be '
    'similar, one of "grl", "mmd", "corr".')

tf.app.flags.DEFINE_string('recon_loss_name', 'sum_of_pairwise_squares',
                           'The name of the reconstruction loss.')

tf.app.flags.DEFINE_string('basic_tower', 'pose_mini',
                           'The basic tower building block.')

def provide_batch_fn():
  """ The provide_batch function to use. """
  return dataset_factory.provide_batch

def main(_):
  model_params = {
      'use_separation': FLAGS.use_separation,
      'domain_separation_startpoint': FLAGS.domain_separation_startpoint,
      'layers_to_regularize': FLAGS.layers_to_regularize,
      'alpha_weight': FLAGS.alpha_weight,
      'beta_weight': FLAGS.beta_weight,
      'gamma_weight': FLAGS.gamma_weight,
      'pose_weight': FLAGS.pose_weight,
      'recon_loss_name': FLAGS.recon_loss_name,
      'decoder_name': FLAGS.decoder_name,
      'encoder_name': FLAGS.encoder_name,
      'weight_decay': FLAGS.weight_decay,
      'batch_size': FLAGS.batch_size,
      'use_logging': FLAGS.use_logging,
      'ps_tasks': FLAGS.ps_tasks,
      'task': FLAGS.task,
  }
  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      # Load the data.
      source_images, source_labels = provide_batch_fn()(
          FLAGS.source_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
          FLAGS.batch_size, FLAGS.num_preprocessing_threads)
      target_images, target_labels = provide_batch_fn()(
          FLAGS.target_dataset, 'train', FLAGS.dataset_dir, FLAGS.num_readers,
          FLAGS.batch_size, FLAGS.num_preprocessing_threads)

      # In the unsupervised case all the samples in the labeled
      # domain are from the source domain.
      domain_selection_mask = tf.fill((source_images.get_shape().as_list()[0],),
                                      True)

      # When using the semisupervised model we include labeled target data in
      # the source labelled data.
      if FLAGS.target_labeled_dataset != 'none':
        # 1000 is the maximum number of labelled target samples that exists in
        # the datasets.
        target_semi_images, target_semi_labels = provide_batch_fn()(
            FLAGS.target_labeled_dataset, 'train', FLAGS.batch_size)

        # Calculate the proportion of source domain samples in the semi-
        # supervised setting, so that the proportion is set accordingly in the
        # batches.
        proportion = float(source_labels['num_train_samples']) / (
            source_labels['num_train_samples'] +
            target_semi_labels['num_train_samples'])

        rnd_tensor = tf.random_uniform(
            (target_semi_images.get_shape().as_list()[0],))

        domain_selection_mask = rnd_tensor < proportion
        source_images = tf.where(domain_selection_mask, source_images,
                                 target_semi_images)
        source_class_labels = tf.where(domain_selection_mask,
                                       source_labels['classes'],
                                       target_semi_labels['classes'])

        if 'quaternions' in source_labels:
          source_pose_labels = tf.where(domain_selection_mask,
                                        source_labels['quaternions'],
                                        target_semi_labels['quaternions'])
          (source_images, source_class_labels, source_pose_labels,
           domain_selection_mask) = tf.train.shuffle_batch(
               [
                   source_images, source_class_labels, source_pose_labels,
                   domain_selection_mask
               ],
               FLAGS.batch_size,
               50000,
               5000,
               num_threads=1,
               enqueue_many=True)

        else:
          (source_images, source_class_labels,
           domain_selection_mask) = tf.train.shuffle_batch(
               [source_images, source_class_labels, domain_selection_mask],
               FLAGS.batch_size,
               50000,
               5000,
               num_threads=1,
               enqueue_many=True)
        source_labels = {}
        source_labels['classes'] = source_class_labels
        if 'quaternions' in source_labels:
          source_labels['quaternions'] = source_pose_labels

      slim.get_or_create_global_step()
      tf.summary.image('source_images', source_images, max_outputs=3)
      tf.summary.image('target_images', target_images, max_outputs=3)

      dsn.create_model(
          source_images,
          source_labels,
          domain_selection_mask,
          target_images,
          target_labels,
          FLAGS.similarity_loss,
          model_params,
          basic_tower_name=FLAGS.basic_tower)

      # Configure the optimization scheme:
      learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate,
          slim.get_or_create_global_step(),
          FLAGS.decay_steps,
          FLAGS.decay_rate,
          staircase=True,
          name='learning_rate')

      tf.summary.scalar('learning_rate', learning_rate)
      tf.summary.scalar('total_loss', tf.losses.get_total_loss())

      opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
      tf.logging.set_verbosity(tf.logging.INFO)
      # Run training.
      loss_tensor = slim.learning.create_train_op(
          slim.losses.get_total_loss(),
          opt,
          summarize_gradients=True,
          colocate_gradients_with_ops=True)
      slim.learning.train(
          train_op=loss_tensor,
          logdir=FLAGS.train_log_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          number_of_steps=FLAGS.max_number_of_steps,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
  tf.app.run()
