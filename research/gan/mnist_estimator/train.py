# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Trains a GANEstimator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import scipy.misc
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from mnist import data_provider
from mnist import networks

tfgan = tf.contrib.gan

flags.DEFINE_integer('batch_size', 32,
                     'The number of images in each train batch.')

flags.DEFINE_integer('max_number_of_steps', 20000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'noise_dims', 64, 'Dimensions of the generator noise vector')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_string('eval_dir', '/tmp/mnist-estimator/',
                    'Directory where the results are saved to.')

FLAGS = flags.FLAGS


def _get_train_input_fn(batch_size, noise_dims, dataset_dir=None,
                        num_threads=4):
  def train_input_fn():
    with tf.device('/cpu:0'):
      images, _, _ = data_provider.provide_data(
          'train', batch_size, dataset_dir, num_threads=num_threads)
    noise = tf.random_normal([batch_size, noise_dims])
    return noise, images
  return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
  def predict_input_fn():
    noise = tf.random_normal([batch_size, noise_dims])
    return noise
  return predict_input_fn


def _unconditional_generator(noise, mode):
  """MNIST generator with extra argument for tf.Estimator's `mode`."""
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  return networks.unconditional_generator(noise, is_training=is_training)


def main(_):
  # Initialize GANEstimator with options and hyperparameters.
  gan_estimator = tfgan.estimator.GANEstimator(
      generator_fn=_unconditional_generator,
      discriminator_fn=networks.unconditional_discriminator,
      generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
      discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
      generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
      discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
      add_summaries=tfgan.estimator.SummaryType.IMAGES)

  # Train estimator.
  train_input_fn = _get_train_input_fn(
      FLAGS.batch_size, FLAGS.noise_dims, FLAGS.dataset_dir)
  gan_estimator.train(train_input_fn, max_steps=FLAGS.max_number_of_steps)

  # Run inference.
  predict_input_fn = _get_predict_input_fn(36, FLAGS.noise_dims)
  prediction_iterable = gan_estimator.predict(predict_input_fn)
  predictions = [prediction_iterable.next() for _ in xrange(36)]

  # Nicely tile.
  image_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in
                range(0, 36, 6)]
  tiled_image = np.concatenate(image_rows, axis=1)

  # Write to disk.
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  scipy.misc.imsave(os.path.join(FLAGS.eval_dir, 'unconditional_gan.png'),
                    np.squeeze(tiled_image, axis=2))


if __name__ == '__main__':
  tf.app.run()
