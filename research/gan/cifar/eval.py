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
"""Evaluates a TFGAN trained CIFAR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data_provider
import networks
import util


flags = tf.flags
FLAGS = tf.flags.FLAGS
tfgan = tf.contrib.gan

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/cifar10/',
                    'Directory where the results are saved to.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('num_images_generated', 100,
                     'Number of images to generate at once.')

flags.DEFINE_integer('num_inception_images', 10,
                     'The number of images to run through Inception at once.')

flags.DEFINE_boolean('eval_real_images', False,
                     'If `True`, run Inception network on real images.')

flags.DEFINE_boolean('conditional_eval', False,
                     'If `True`, set up a conditional GAN.')

flags.DEFINE_boolean('eval_frechet_inception_distance', True,
                     'If `True`, compute Frechet Inception distance using real '
                     'images and generated images.')

flags.DEFINE_integer('num_images_per_class', 10,
                     'When a conditional generator is used, this is the number '
                     'of images to display per class.')

flags.DEFINE_integer('max_number_of_evaluations', None,
                     'Number of times to run evaluation. If `None`, run '
                     'forever.')


def main(_, run_eval_loop=True):
  # Fetch and generate images to run through Inception.
  with tf.name_scope('inputs'):
    real_data, num_classes = _get_real_data(
        FLAGS.num_images_generated, FLAGS.dataset_dir)
    generated_data = _get_generated_data(
        FLAGS.num_images_generated, FLAGS.conditional_eval, num_classes)

  # Compute Frechet Inception Distance.
  if FLAGS.eval_frechet_inception_distance:
    fid = util.get_frechet_inception_distance(
        real_data, generated_data, FLAGS.num_images_generated,
        FLAGS.num_inception_images)
    tf.summary.scalar('frechet_inception_distance', fid)

  # Compute normal Inception scores.
  if FLAGS.eval_real_images:
    inc_score = util.get_inception_scores(
        real_data, FLAGS.num_images_generated, FLAGS.num_inception_images)
  else:
    inc_score = util.get_inception_scores(
        generated_data, FLAGS.num_images_generated, FLAGS.num_inception_images)
  tf.summary.scalar('inception_score', inc_score)

  # If conditional, display an image grid of difference classes.
  if FLAGS.conditional_eval and not FLAGS.eval_real_images:
    reshaped_imgs = util.get_image_grid(
        generated_data, FLAGS.num_images_generated, num_classes,
        FLAGS.num_images_per_class)
    tf.summary.image('generated_data', reshaped_imgs, max_outputs=1)

  # Create ops that write images to disk.
  image_write_ops = None
  if FLAGS.conditional_eval:
    reshaped_imgs = util.get_image_grid(
        generated_data, FLAGS.num_images_generated, num_classes,
        FLAGS.num_images_per_class)
    uint8_images = data_provider.float_image_to_uint8(reshaped_imgs)
    image_write_ops = tf.write_file(
        '%s/%s'% (FLAGS.eval_dir, 'conditional_cifar10.png'),
        tf.image.encode_png(uint8_images[0]))
  else:
    if FLAGS.num_images_generated >= 100:
      reshaped_imgs = tfgan.eval.image_reshaper(
          generated_data[:100], num_cols=FLAGS.num_images_per_class)
      uint8_images = data_provider.float_image_to_uint8(reshaped_imgs)
      image_write_ops = tf.write_file(
          '%s/%s'% (FLAGS.eval_dir, 'unconditional_cifar10.png'),
          tf.image.encode_png(uint8_images[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop: return
  tf.contrib.training.evaluate_repeatedly(
      FLAGS.checkpoint_dir,
      master=FLAGS.master,
      hooks=[tf.contrib.training.SummaryAtEndHook(FLAGS.eval_dir),
             tf.contrib.training.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=FLAGS.max_number_of_evaluations)


def _get_real_data(num_images_generated, dataset_dir):
  """Get real images."""
  data, _, _, num_classes = data_provider.provide_data(
      num_images_generated, dataset_dir)
  return data, num_classes


def _get_generated_data(num_images_generated, conditional_eval, num_classes):
  """Get generated images."""
  noise = tf.random_normal([num_images_generated, 64])
  # If conditional, generate class-specific images.
  if conditional_eval:
    conditioning = util.get_generator_conditioning(
        num_images_generated, num_classes)
    generator_inputs = (noise, conditioning)
    generator_fn = networks.conditional_generator
  else:
    generator_inputs = noise
    generator_fn = networks.generator
  # In order for variables to load, use the same variable scope as in the
  # train job.
  with tf.variable_scope('Generator'):
    data = generator_fn(generator_inputs)

  return data


if __name__ == '__main__':
  tf.app.run()
