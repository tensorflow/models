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
"""Evaluates a TFGAN trained MNIST model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import data_provider
import networks
import util

flags = tf.flags
FLAGS = flags.FLAGS
tfgan = tf.contrib.gan


flags.DEFINE_string('checkpoint_dir', '/tmp/mnist/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/mnist/',
                    'Directory where the results are saved to.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

flags.DEFINE_integer('num_images_generated', 1000,
                     'Number of images to generate at once.')

flags.DEFINE_boolean('eval_real_images', False,
                     'If `True`, run Inception network on real images.')

flags.DEFINE_integer('noise_dims', 64,
                     'Dimensions of the generator noise vector')

flags.DEFINE_string('classifier_filename', None,
                    'Location of the pretrained classifier. If `None`, use '
                    'default.')

flags.DEFINE_integer('max_number_of_evaluations', None,
                     'Number of times to run evaluation. If `None`, run '
                     'forever.')


def main(_, run_eval_loop=True):
  # Fetch real images.
  with tf.name_scope('inputs'):
    real_images, _, _ = data_provider.provide_data(
        'train', FLAGS.num_images_generated, FLAGS.dataset_dir)

  image_write_ops = None
  if FLAGS.eval_real_images:
    tf.summary.scalar('MNIST_Classifier_score',
                      util.mnist_score(real_images, FLAGS.classifier_filename))
  else:
    # In order for variables to load, use the same variable scope as in the
    # train job.
    with tf.variable_scope('Generator'):
      images = networks.unconditional_generator(
          tf.random_normal([FLAGS.num_images_generated, FLAGS.noise_dims]))
    tf.summary.scalar('MNIST_Frechet_distance',
                      util.mnist_frechet_distance(
                          real_images, images, FLAGS.classifier_filename))
    tf.summary.scalar('MNIST_Classifier_score',
                      util.mnist_score(images, FLAGS.classifier_filename))
    if FLAGS.num_images_generated >= 100:
      reshaped_images = tfgan.eval.image_reshaper(
          images[:100, ...], num_cols=10)
      uint8_images = data_provider.float_image_to_uint8(reshaped_images)
      image_write_ops = tf.write_file(
          '%s/%s'% (FLAGS.eval_dir, 'unconditional_gan.png'),
          tf.image.encode_png(uint8_images[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop: return
  tf.contrib.training.evaluate_repeatedly(
      FLAGS.checkpoint_dir,
      hooks=[tf.contrib.training.SummaryAtEndHook(FLAGS.eval_dir),
             tf.contrib.training.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=FLAGS.max_number_of_evaluations)


if __name__ == '__main__':
  tf.app.run()
