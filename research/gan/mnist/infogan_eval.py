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
"""Evaluates an InfoGAN TFGAN trained MNIST model.

The image visualizations, as in https://arxiv.org/abs/1606.03657, show the
effect of varying a specific latent variable on the image. Each visualization
focuses on one of the three structured variables. Columns have two of the three
variables fixed, while the third one is varied. Different rows have different
random samples from the remaining latents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
import numpy as np

import tensorflow as tf

import data_provider
import networks
import util

tfgan = tf.contrib.gan


flags.DEFINE_string('checkpoint_dir', '/tmp/mnist/',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/mnist/',
                    'Directory where the results are saved to.')

flags.DEFINE_integer('noise_samples', 6,
                     'Number of samples to draw from the continuous structured '
                     'noise.')

flags.DEFINE_integer('unstructured_noise_dims', 62,
                     'The number of dimensions of the unstructured noise.')

flags.DEFINE_integer('continuous_noise_dims', 2,
                     'The number of dimensions of the continuous noise.')

flags.DEFINE_string('classifier_filename', None,
                    'Location of the pretrained classifier. If `None`, use '
                    'default.')

flags.DEFINE_integer('max_number_of_evaluations', None,
                     'Number of times to run evaluation. If `None`, run '
                     'forever.')

flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')

CAT_SAMPLE_POINTS = np.arange(0, 10)
CONT_SAMPLE_POINTS = np.linspace(-2.0, 2.0, 10)
FLAGS = flags.FLAGS


def main(_, run_eval_loop=True):
  with tf.name_scope('inputs'):
    noise_args = (FLAGS.noise_samples, CAT_SAMPLE_POINTS, CONT_SAMPLE_POINTS,
                  FLAGS.unstructured_noise_dims, FLAGS.continuous_noise_dims)
    # Use fixed noise vectors to illustrate the effect of each dimension.
    display_noise1 = util.get_eval_noise_categorical(*noise_args)
    display_noise2 = util.get_eval_noise_continuous_dim1(*noise_args)
    display_noise3 = util.get_eval_noise_continuous_dim2(*noise_args)
    _validate_noises([display_noise1, display_noise2, display_noise3])

  # Visualize the effect of each structured noise dimension on the generated
  # image.
  def generator_fn(inputs):
    return networks.infogan_generator(
        inputs, len(CAT_SAMPLE_POINTS), is_training=False)
  with tf.variable_scope('Generator') as genscope:  # Same scope as in training.
    categorical_images = generator_fn(display_noise1)
  reshaped_categorical_img = tfgan.eval.image_reshaper(
      categorical_images, num_cols=len(CAT_SAMPLE_POINTS))
  tf.summary.image('categorical', reshaped_categorical_img, max_outputs=1)

  with tf.variable_scope(genscope, reuse=True):
    continuous1_images = generator_fn(display_noise2)
  reshaped_continuous1_img = tfgan.eval.image_reshaper(
      continuous1_images, num_cols=len(CONT_SAMPLE_POINTS))
  tf.summary.image('continuous1', reshaped_continuous1_img, max_outputs=1)

  with tf.variable_scope(genscope, reuse=True):
    continuous2_images = generator_fn(display_noise3)
  reshaped_continuous2_img = tfgan.eval.image_reshaper(
      continuous2_images, num_cols=len(CONT_SAMPLE_POINTS))
  tf.summary.image('continuous2', reshaped_continuous2_img, max_outputs=1)

  # Evaluate image quality.
  all_images = tf.concat(
      [categorical_images, continuous1_images, continuous2_images], 0)
  tf.summary.scalar('MNIST_Classifier_score',
                    util.mnist_score(all_images, FLAGS.classifier_filename))

  # Write images to disk.
  image_write_ops = []
  if FLAGS.write_to_disk:
    image_write_ops.append(_get_write_image_ops(
        FLAGS.eval_dir, 'categorical_infogan.png', reshaped_categorical_img[0]))
    image_write_ops.append(_get_write_image_ops(
        FLAGS.eval_dir, 'continuous1_infogan.png', reshaped_continuous1_img[0]))
    image_write_ops.append(_get_write_image_ops(
        FLAGS.eval_dir, 'continuous2_infogan.png', reshaped_continuous2_img[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop: return
  tf.contrib.training.evaluate_repeatedly(
      FLAGS.checkpoint_dir,
      hooks=[tf.contrib.training.SummaryAtEndHook(FLAGS.eval_dir),
             tf.contrib.training.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=FLAGS.max_number_of_evaluations)


def _validate_noises(noises):
  """Sanity check on constructed noise tensors.

  Args:
    noises: List of 3-tuples of noise vectors.
  """
  assert isinstance(noises, (list, tuple))
  for noise_l in noises:
    assert len(noise_l) == 3
    assert isinstance(noise_l[0], np.ndarray)
    batch_dim = noise_l[0].shape[0]
    for i, noise in enumerate(noise_l):
      assert isinstance(noise, np.ndarray)
      # Check that batch dimensions are all the same.
      assert noise.shape[0] == batch_dim

      # Check that shapes for corresponding noises are the same.
      assert noise.shape == noises[0][i].shape


def _get_write_image_ops(eval_dir, filename, images):
  """Create Ops that write images to disk."""
  return tf.write_file(
      '%s/%s'% (eval_dir, filename),
      tf.image.encode_png(data_provider.float_image_to_uint8(images)))


if __name__ == '__main__':
  app.run(main)
