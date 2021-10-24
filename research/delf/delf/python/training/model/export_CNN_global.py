# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Export global CNN feature tensorflow inference model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import tensorflow as tf

from delf.python.training.model import global_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', None, help='Path to saved checkpoint.')
flags.DEFINE_string('export_path', None,
                    help='Path where model will be exported.')
flags.DEFINE_list(
  'input_scales_list', None,
  'Optional input image scales to use. If None (default), an input end-point '
  '"input_scales" is added for the exported model. If not None, the '
  'specified list of floats will be hard-coded as the desired input scales.')
flags.DEFINE_enum(
  'multi_scale_pool_type', 'None', ['None', 'average', 'sum'],
  "If 'None' (default), the model is exported with an output end-point "
  "'global_descriptors', where the global descriptor for each scale is "
  "returned separately. If not 'None', the global descriptor of each scale is"
  ' pooled and a 1D global descriptor is returned, with output end-point '
  "'global_descriptor'.")
flags.DEFINE_boolean('normalize_global_descriptor', False,
                     'If True, L2-normalizes global descriptor.')
# network architecture and initialization options
flags.DEFINE_string('arch', 'ResNet101',
                    'model architecture (default: ResNet101)')
flags.DEFINE_string('pool', 'gem', 'pooling options (default: gem)')
flags.DEFINE_boolean('whitening', False,
                     'train model with learnable whitening (linear layer) '
                     'after the pooling')


def NormalizeImages(images):
  """Normalize pixel values in image.

  Args:
    images: `Tensor`, images to normalize.

  Returns:
    normalized_images: `Tensor`, normalized images.
  """
  preprocess_input(images, mode='caffe')
  return images


@tf.function
def ExtractGlobalFeatures(image,
                          image_scales,
                          global_scales_ind,
                          model_fn,
                          multi_scale_pool_type='None',
                          normalize_global_descriptor=False):
  """Extract global features for input image.

  Args:
    image: image tensor of type tf.uint8 with shape [h, w, channels].
    image_scales: 1D float tensor which contains float scales used for image
      pyramid construction.
    global_scales_ind: Feature extraction happens only for a subset of
      `image_scales`, those with corresponding indices from this tensor.
    model_fn: model function. Follows the signature:
      * Args:
        * `images`: Image tensor which is re-scaled.
      * Returns:
        * `global_descriptors`: Global descriptors for input images.
    multi_scale_pool_type: If set, the global descriptor of each scale is pooled
      and a 1D global descriptor is returned.
    normalize_global_descriptor: If True, output global descriptors are
      L2-normalized.

  Returns:
    global_descriptors: If `multi_scale_pool_type` is 'None', returns a [S, D]
      float tensor. S is the number of scales, and D the global descriptor
      dimensionality. Each D-dimensional entry is a global descriptor, which may
      be L2-normalized depending on `normalize_global_descriptor`. If
      `multi_scale_pool_type` is not 'None', returns a [D] float tensor with the
      pooled global descriptor.

  """
  original_image_shape_float = tf.gather(
    tf.dtypes.cast(tf.shape(image), tf.float32), [0, 1])
  image_tensor = NormalizeImages(image)
  image_tensor = tf.expand_dims(image_tensor, 0, name='image/expand_dims')

  def _ResizeAndExtract(scale_index):
    """Helper function to resize image then extract global feature.

    Args:
      scale_index: A valid index in image_scales.

    Returns:
      global_descriptor: [1,D] tensor denoting the extracted global descriptor.
    """
    scale = tf.gather(image_scales, scale_index)
    new_image_size = tf.dtypes.cast(
      tf.round(original_image_shape_float * scale), tf.int32)
    resized_image = tf.image.resize(image_tensor, new_image_size)
    global_descriptor = model_fn(resized_image)
    return global_descriptor

  # First loop to find initial scale to be used.
  num_scales = tf.shape(image_scales)[0]
  initial_scale_index = tf.constant(-1, dtype=tf.int32)
  for scale_index in tf.range(num_scales):
    if tf.reduce_any(tf.equal(global_scales_ind, scale_index)):
      initial_scale_index = scale_index
      break

  output_global = _ResizeAndExtract(initial_scale_index)

  # Loop over subsequent scales.
  for scale_index in tf.range(initial_scale_index + 1, num_scales):
    # Allow an undefined number of global feature scales to be extracted.
    tf.autograph.experimental.set_loop_options(
      shape_invariants=[(output_global, tf.TensorShape([None, None]))])

    if tf.reduce_any(tf.equal(global_scales_ind, scale_index)):
      global_descriptor = _ResizeAndExtract(scale_index)
      output_global = tf.concat([output_global, global_descriptor], 0)

  normalization_axis = 1
  if multi_scale_pool_type == 'average':
    output_global = tf.reduce_mean(
      output_global,
      axis=0,
      keepdims=False,
      name='multi_scale_average_pooling')
    normalization_axis = 0
  elif multi_scale_pool_type == 'sum':
    output_global = tf.reduce_sum(
      output_global, axis=0, keepdims=False,
      name='multi_scale_sum_pooling')
    normalization_axis = 0

  if normalize_global_descriptor:
    output_global = tf.nn.l2_normalize(
      output_global, axis=normalization_axis, name='l2_normalization')

  return output_global


class _ExtractModule(tf.Module):
  """Helper module to build and save global feature model."""

  def __init__(self,
               multi_scale_pool_type='None',
               normalize_global_descriptor=False,
               input_scales_tensor=None):
    """Initialization of global feature model.
    Args:
      multi_scale_pool_type: Type of multi-scale pooling to perform.
      normalize_global_descriptor: Whether to L2-normalize global
        descriptor.
      input_scales_tensor: If None, the exported function to be used
        should be ExtractFeatures, where an input end-point "input_scales" is
        added for the exported model. If not None, the specified 1D tensor of
        floats will be hard-coded as the desired input scales, in conjunction
         with ExtractFeaturesFixedScales.
    """
    self._multi_scale_pool_type = multi_scale_pool_type
    self._normalize_global_descriptor = normalize_global_descriptor
    if input_scales_tensor is None:
      self._input_scales_tensor = []
    else:
      self._input_scales_tensor = input_scales_tensor

    self._model = global_model.GlobalFeatureNet(FLAGS.arch, FLAGS.pool,
                                            FLAGS.whitening, pretrained=False)

  def LoadWeights(self, checkpoint_path):
    self._model.load_weights(checkpoint_path)

  @tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8,
                  name='input_image'),
    tf.TensorSpec(shape=[None], dtype=tf.float32, name='input_scales'),
    tf.TensorSpec(shape=[None], dtype=tf.int32,
                  name='input_global_scales_ind')
  ])
  def ExtractFeatures(self, input_image, input_scales,
                      input_global_scales_ind):
    extracted_features = ExtractGlobalFeatures(
      input_image,
      input_scales,
      input_global_scales_ind,
      lambda x: self._model(x, training=False),
      multi_scale_pool_type=self._multi_scale_pool_type,
      normalize_global_descriptor=self._normalize_global_descriptor)

    named_output_tensors = {}
    named_output_tensors['global_descriptors'] = tf.identity(
      extracted_features, name='global_descriptors')
    return named_output_tensors

  @tf.function(input_signature=[
    tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
  ])
  def ExtractFeaturesFixedScales(self, input_image):
    exp_image = tf.expand_dims(input_image, axis=0)
    return self._model(exp_image, training=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  export_path = FLAGS.export_path
  if os.path.exists(export_path):
    raise ValueError('export_path %s already exists.' % export_path)

  if FLAGS.input_scales_list is None:
    input_scales_tensor = None
  else:
    input_scales_tensor = tf.constant(
      [float(s) for s in FLAGS.input_scales_list],
      dtype=tf.float32,
      shape=[len(FLAGS.input_scales_list)],
      name='input_scales')
  module = _ExtractModule(FLAGS.multi_scale_pool_type,
                          FLAGS.normalize_global_descriptor,
                          input_scales_tensor)

  # Load the weights.
  checkpoint_path = FLAGS.ckpt_path
  module.LoadWeights(checkpoint_path)
  print('Checkpoint loaded from ', checkpoint_path)

  # Save the module.
  if FLAGS.input_scales_list is None:
    served_function = module.ExtractFeatures
  else:
    served_function = module.ExtractFeaturesFixedScales

  tf.saved_model.save(
    module, export_path, signatures={'serving_default': served_function})


if __name__ == '__main__':
  app.run(main)
