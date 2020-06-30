# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Export global feature tensorflow inference model.

This model includes image pyramids for multi-scale processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import tensorflow as tf

from delf.python.training.model import delf_model
from delf.python.training.model import export_model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', '/tmp/delf-logdir/delf-weights',
                    'Path to saved checkpoint.')
flags.DEFINE_string('export_path', None, 'Path where model will be exported.')
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


class _ExtractModule(tf.Module):
  """Helper module to build and save global feature model."""

  def __init__(self,
               multi_scale_pool_type='None',
               normalize_global_descriptor=False,
               input_scales_tensor=None):
    """Initialization of global feature model.

    Args:
      multi_scale_pool_type: Type of multi-scale pooling to perform.
      normalize_global_descriptor: Whether to L2-normalize global descriptor.
      input_scales_tensor: If None, the exported function to be used should be
        ExtractFeatures, where an input end-point "input_scales" is added for
        the exported model. If not None, the specified 1D tensor of floats will
        be hard-coded as the desired input scales, in conjunction with
        ExtractFeaturesFixedScales.
    """
    self._multi_scale_pool_type = multi_scale_pool_type
    self._normalize_global_descriptor = normalize_global_descriptor
    if input_scales_tensor is None:
      self._input_scales_tensor = []
    else:
      self._input_scales_tensor = input_scales_tensor

    # Setup the DELF model for extraction.
    self._model = delf_model.Delf(block3_strides=False, name='DELF')

  def LoadWeights(self, checkpoint_path):
    self._model.load_weights(checkpoint_path)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image'),
      tf.TensorSpec(shape=[None], dtype=tf.float32, name='input_scales'),
      tf.TensorSpec(
          shape=[None], dtype=tf.int32, name='input_global_scales_ind')
  ])
  def ExtractFeatures(self, input_image, input_scales, input_global_scales_ind):
    extracted_features = export_model_utils.ExtractGlobalFeatures(
        input_image,
        input_scales,
        input_global_scales_ind,
        lambda x: self._model.backbone.build_call(x, training=False),
        multi_scale_pool_type=self._multi_scale_pool_type,
        normalize_global_descriptor=self._normalize_global_descriptor)

    named_output_tensors = {}
    if self._multi_scale_pool_type == 'None':
      named_output_tensors['global_descriptors'] = tf.identity(
          extracted_features, name='global_descriptors')
    else:
      named_output_tensors['global_descriptor'] = tf.identity(
          extracted_features, name='global_descriptor')

    return named_output_tensors

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
  ])
  def ExtractFeaturesFixedScales(self, input_image):
    return self.ExtractFeatures(input_image, self._input_scales_tensor,
                                tf.range(tf.size(self._input_scales_tensor)))


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

  # Save the module
  if FLAGS.input_scales_list is None:
    served_function = module.ExtractFeatures
  else:
    served_function = module.ExtractFeaturesFixedScales

  tf.saved_model.save(
      module, export_path, signatures={'serving_default': served_function})


if __name__ == '__main__':
  app.run(main)
