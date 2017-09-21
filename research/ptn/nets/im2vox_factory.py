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

"""Factory module for getting the complete image to voxel generation network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import perspective_projector
from nets import ptn_encoder
from nets import ptn_vox_decoder

_NAME_TO_NETS = {
    'ptn_encoder': ptn_encoder,
    'ptn_vox_decoder': ptn_vox_decoder,
    'perspective_projector': perspective_projector,
}


def _get_network(name):
  """Gets a single encoder/decoder network model."""

  if name not in _NAME_TO_NETS:
    raise ValueError('Network name [%s] not recognized.' % name)
  return _NAME_TO_NETS[name].model


def get(params, is_training=False, reuse=False, run_projection=True):
  """Factory function to get the training/pretraining im->vox model (NIPS16).

  Args:
    params: Different parameters used througout ptn, typically FLAGS (dict).
    is_training: Set to True if while training (boolean).
    reuse: Set as True if sharing variables with a model that has already
      been built (boolean).
    run_projection: Set as False if not interested in mask and projection
      images. Useful in evaluation routine (boolean).
  Returns:
    Model function for network (inputs to outputs).
  """
  def model(inputs):
    """Model function corresponding to a specific network architecture."""
    outputs = {}

    # First, build the encoder
    encoder_fn = _get_network(params.encoder_name)
    with tf.variable_scope('encoder', reuse=reuse):
      # Produces id/pose units
      enc_outputs = encoder_fn(inputs['images_1'], params, is_training)
      outputs['ids_1'] = enc_outputs['ids']

    # Second, build the decoder and projector
    decoder_fn = _get_network(params.decoder_name)
    with tf.variable_scope('decoder', reuse=reuse):
      outputs['voxels_1'] = decoder_fn(outputs['ids_1'], params, is_training)
    if run_projection:
      projector_fn = _get_network(params.projector_name)
      with tf.variable_scope('projector', reuse=reuse):
        outputs['projs_1'] = projector_fn(
            outputs['voxels_1'], inputs['matrix_1'], params, is_training)
      # Infer the ground-truth mask
      with tf.variable_scope('oracle', reuse=reuse):
        outputs['masks_1'] = projector_fn(inputs['voxels'], inputs['matrix_1'],
                                          params, False)

      # Third, build the entire graph (bundled strategy described in PTN paper)
      for k in range(1, params.step_size):
        with tf.variable_scope('projector', reuse=True):
          outputs['projs_%d' % (k + 1)] = projector_fn(
              outputs['voxels_1'], inputs['matrix_%d' %
                                          (k + 1)], params, is_training)
        with tf.variable_scope('oracle', reuse=True):
          outputs['masks_%d' % (k + 1)] = projector_fn(
              inputs['voxels'], inputs['matrix_%d' % (k + 1)], params, False)

    return outputs

  return model
