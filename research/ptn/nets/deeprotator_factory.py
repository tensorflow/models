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

"""Factory module for different encoder/decoder network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import ptn_encoder
from nets import ptn_im_decoder
from nets import ptn_rotator

_NAME_TO_NETS = {
    'ptn_encoder': ptn_encoder,
    'ptn_rotator': ptn_rotator,
    'ptn_im_decoder': ptn_im_decoder,
}


def _get_network(name):
  """Gets a single network component."""

  if name not in _NAME_TO_NETS:
    raise ValueError('Network name [%s] not recognized.' % name)
  return _NAME_TO_NETS[name].model


def get(params, is_training=False, reuse=False):
  """Factory function to retrieve a network model.

  Args:
    params: Different parameters used througout ptn, typically FLAGS (dict)
    is_training: Set to True if while training (boolean)
    reuse: Set as True if either using a pre-trained model or
      in the training loop while the graph has already been built (boolean)
  Returns:
    Model function for network (inputs to outputs)
  """

  def model(inputs):
    """Model function corresponding to a specific network architecture."""
    outputs = {}

    # First, build the encoder.
    encoder_fn = _get_network(params.encoder_name)
    with tf.variable_scope('encoder', reuse=reuse):
      # Produces id/pose units
      features = encoder_fn(inputs['images_0'], params, is_training)
      outputs['ids'] = features['ids']
      outputs['poses_0'] = features['poses']

    # Second, build the rotator and decoder.
    rotator_fn = _get_network(params.rotator_name)
    with tf.variable_scope('rotator', reuse=reuse):
      outputs['poses_1'] = rotator_fn(outputs['poses_0'], inputs['actions'],
                                      params, is_training)
    decoder_fn = _get_network(params.decoder_name)
    with tf.variable_scope('decoder', reuse=reuse):
      dec_output = decoder_fn(outputs['ids'], outputs['poses_1'], params,
                              is_training)
      outputs['images_1'] = dec_output['images']
      outputs['masks_1'] = dec_output['masks']

    # Third, build the recurrent connection
    for k in range(1, params.step_size):
      with tf.variable_scope('rotator', reuse=True):
        outputs['poses_%d' % (k + 1)] = rotator_fn(
            outputs['poses_%d' % k], inputs['actions'], params, is_training)
      with tf.variable_scope('decoder', reuse=True):
        dec_output = decoder_fn(outputs['ids'], outputs['poses_%d' % (k + 1)],
                                params, is_training)
        outputs['images_%d' % (k + 1)] = dec_output['images']
        outputs['masks_%d' % (k + 1)] = dec_output['masks']

    return outputs

  return model
