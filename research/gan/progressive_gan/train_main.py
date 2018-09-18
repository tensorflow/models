# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train a progressive GAN model.

See https://arxiv.org/abs/1710.10196 for details about the model.

See https://github.com/tkarras/progressive_growing_of_gans for the original
theano implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


from absl import flags
from absl import logging
import tensorflow as tf

import data_provider
import train

tfgan = tf.contrib.gan

flags.DEFINE_string('dataset_name', 'cifar10', 'Dataset name.')

flags.DEFINE_string('dataset_file_pattern', '', 'Dataset file pattern.')

flags.DEFINE_integer('start_height', 4, 'Start image height.')

flags.DEFINE_integer('start_width', 4, 'Start image width.')

flags.DEFINE_integer('scale_base', 2, 'Resolution multiplier.')

flags.DEFINE_integer('num_resolutions', 4, 'Number of progressive resolutions.')

flags.DEFINE_integer('kernel_size', 3, 'Convolution kernel size.')

flags.DEFINE_integer('colors', 3, 'Number of image channels.')

flags.DEFINE_bool('to_rgb_use_tanh_activation', False,
                  'Whether to apply tanh activation when output rgb.')

flags.DEFINE_integer('batch_size', 8, 'Number of images in each batch.')

flags.DEFINE_integer('stable_stage_num_images', 1000,
                     'Number of images in the stable stage.')

flags.DEFINE_integer('transition_stage_num_images', 1000,
                     'Number of images in the transition stage.')

flags.DEFINE_integer('total_num_images', 10000, 'Total number of images.')

flags.DEFINE_integer('save_summaries_num_images', 100,
                     'Save summaries in this number of images.')

flags.DEFINE_integer('latent_vector_size', 128, 'Latent vector size.')

flags.DEFINE_integer('fmap_base', 4096, 'Base number of filters.')

flags.DEFINE_float('fmap_decay', 1.0, 'Decay of number of filters.')

flags.DEFINE_integer('fmap_max', 128, 'Max number of filters.')

flags.DEFINE_float('gradient_penalty_target', 1.0,
                   'Gradient norm target for wasserstein loss.')

flags.DEFINE_float('gradient_penalty_weight', 10.0,
                   'Gradient penalty weight for wasserstein loss.')

flags.DEFINE_float('real_score_penalty_weight', 0.001,
                   'Additional penalty to keep the scores from drifting too '
                   'far from zero.')

flags.DEFINE_float('generator_learning_rate', 0.001, 'Learning rate.')

flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Learning rate.')

flags.DEFINE_float('adam_beta1', 0.0, 'Adam beta 1.')

flags.DEFINE_float('adam_beta2', 0.99, 'Adam beta 2.')

flags.DEFINE_integer('fake_grid_size', 8, 'The fake image grid size for eval.')

flags.DEFINE_integer('interp_grid_size', 8,
                     'The interp image grid size for eval.')

flags.DEFINE_string('train_root_dir', '/tmp/progressive_gan/',
                    'Directory where to write event logs.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = flags.FLAGS


def _make_config_from_flags():
  """Makes a config dictionary from commandline flags."""
  return dict([(flag.name, flag.value)
               for flag in FLAGS.get_key_flags_for_module(sys.argv[0])])


def _provide_real_images(**kwargs):
  """Provides real images."""
  dataset_name = kwargs.get('dataset_name')
  dataset_file_pattern = kwargs.get('dataset_file_pattern')
  batch_size = kwargs['batch_size']
  colors = kwargs['colors']
  final_height, final_width = train.make_resolution_schedule(
      **kwargs).final_resolutions
  if dataset_name is not None:
    return data_provider.provide_data(
        dataset_name=dataset_name,
        split_name='train',
        batch_size=batch_size,
        patch_height=final_height,
        patch_width=final_width,
        colors=colors)
  elif dataset_file_pattern is not None:
    return data_provider.provide_data_from_image_files(
        file_pattern=dataset_file_pattern,
        batch_size=batch_size,
        patch_height=final_height,
        patch_width=final_width,
        colors=colors)


def main(_):
  if not tf.gfile.Exists(FLAGS.train_root_dir):
    tf.gfile.MakeDirs(FLAGS.train_root_dir)

  config = _make_config_from_flags()
  logging.info('\n'.join(['{}={}'.format(k, v) for k, v in config.iteritems()]))

  for stage_id in train.get_stage_ids(**config):
    tf.reset_default_graph()
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      real_images = None
      with tf.device('/cpu:0'), tf.name_scope('inputs'):
        real_images = _provide_real_images(**config)
      model = train.build_model(stage_id, real_images, **config)
      train.add_model_summaries(model, **config)
      train.train(model, **config)


if __name__ == '__main__':
  tf.app.run()
