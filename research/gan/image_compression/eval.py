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
"""Evaluates a TFGAN trained compression model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import data_provider
import networks
import summaries

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '/tmp/compression/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/compression/',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('max_number_of_evaluations', None,
                     'Number of times to run evaluation. If `None`, run '
                     'forever.')

flags.DEFINE_string('dataset_dir', None, 'Location of data.')

# Compression-specific flags.
flags.DEFINE_integer('batch_size', 32, 'The number of images in each batch.')

flags.DEFINE_integer('patch_size', 32, 'The size of the patches to train on.')

flags.DEFINE_integer('bits_per_patch', 1230,
                     'The number of bits to produce per patch.')

flags.DEFINE_integer('model_depth', 64,
                     'Number of filters for compression model')


def main(_, run_eval_loop=True):
  with tf.name_scope('inputs'):
    images = data_provider.provide_data(
        'validation', FLAGS.batch_size, dataset_dir=FLAGS.dataset_dir,
        patch_size=FLAGS.patch_size)

  # In order for variables to load, use the same variable scope as in the
  # train job.
  with tf.variable_scope('generator'):
    reconstructions, _, prebinary = networks.compression_model(
        images,
        num_bits=FLAGS.bits_per_patch,
        depth=FLAGS.model_depth,
        is_training=False)
  summaries.add_reconstruction_summaries(images, reconstructions, prebinary)

  # Visualize losses.
  pixel_loss_per_example = tf.reduce_mean(
      tf.abs(images - reconstructions), axis=[1, 2, 3])
  pixel_loss = tf.reduce_mean(pixel_loss_per_example)
  tf.summary.histogram('pixel_l1_loss_hist', pixel_loss_per_example)
  tf.summary.scalar('pixel_l1_loss', pixel_loss)

  # Create ops to write images to disk.
  uint8_images = data_provider.float_image_to_uint8(images)
  uint8_reconstructions = data_provider.float_image_to_uint8(reconstructions)
  uint8_reshaped = summaries.stack_images(uint8_images, uint8_reconstructions)
  image_write_ops = tf.write_file(
      '%s/%s'% (FLAGS.eval_dir, 'compression.png'),
      tf.image.encode_png(uint8_reshaped[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop: return
  tf.contrib.training.evaluate_repeatedly(
      FLAGS.checkpoint_dir,
      master=FLAGS.master,
      hooks=[tf.contrib.training.SummaryAtEndHook(FLAGS.eval_dir),
             tf.contrib.training.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=FLAGS.max_number_of_evaluations)


if __name__ == '__main__':
  tf.app.run()
