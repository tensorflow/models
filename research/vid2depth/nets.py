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

"""Depth and Ego-Motion networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import numpy as np
import tensorflow as tf
import util

slim = tf.contrib.slim

# TODO(rezama): Move flag to main, pass as argument to functions.
flags.DEFINE_bool('use_bn', True, 'Add batch norm layers.')
FLAGS = flags.FLAGS

# Weight regularization.
WEIGHT_REG = 0.05

# Disparity (inverse depth) values range from 0.01 to 10.
DISP_SCALING = 10
MIN_DISP = 0.01

EGOMOTION_VEC_SIZE = 6


def egomotion_net(image_stack, is_training=True, legacy_mode=False):
  """Predict ego-motion vectors from a stack of frames.

  Args:
    image_stack: Input tensor with shape [B, h, w, seq_length * 3].  Regardless
        of the value of legacy_mode, the input image sequence passed to the
        function should be in normal order, e.g. [1, 2, 3].
    is_training: Whether the model is being trained or not.
    legacy_mode: Setting legacy_mode to True enables compatibility with
        SfMLearner checkpoints.  When legacy_mode is on, egomotion_net()
        rearranges the input tensor to place the target (middle) frame first in
        sequence.  This is the arrangement of inputs that legacy models have
        received during training.  In legacy mode, the client program
        (model.Model.build_loss()) interprets the outputs of this network
        differently as well.  For example:

        When legacy_mode == True,
        Network inputs will be [2, 1, 3]
        Network outputs will be [1 -> 2, 3 -> 2]

        When legacy_mode == False,
        Network inputs will be [1, 2, 3]
        Network outputs will be [1 -> 2, 2 -> 3]

  Returns:
    Egomotion vectors with shape [B, seq_length - 1, 6].
  """
  seq_length = image_stack.get_shape()[3].value // 3  # 3 == RGB.
  if legacy_mode:
    # Put the target frame at the beginning of stack.
    with tf.name_scope('rearrange_stack'):
      mid_index = util.get_seq_middle(seq_length)
      left_subset = image_stack[:, :, :, :mid_index * 3]
      target_frame = image_stack[:, :, :, mid_index * 3:(mid_index + 1) * 3]
      right_subset = image_stack[:, :, :, (mid_index + 1) * 3:]
      image_stack = tf.concat([target_frame, left_subset, right_subset], axis=3)
  batch_norm_params = {'is_training': is_training}
  num_egomotion_vecs = seq_length - 1
  with tf.variable_scope('pose_exp_net') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    normalizer_fn = slim.batch_norm if FLAGS.use_bn else None
    normalizer_params = batch_norm_params if FLAGS.use_bn else None
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=normalizer_fn,
                        weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                        normalizer_params=normalizer_params,
                        activation_fn=tf.nn.relu,
                        outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
      cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
      cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
      cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
      cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')

      # Ego-motion specific layers
      with tf.variable_scope('pose'):
        cnv6 = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
        cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
        pred_channels = EGOMOTION_VEC_SIZE * num_egomotion_vecs
        egomotion_pred = slim.conv2d(cnv7,
                                     pred_channels,
                                     [1, 1],
                                     scope='pred',
                                     stride=1,
                                     normalizer_fn=None,
                                     activation_fn=None)
        egomotion_avg = tf.reduce_mean(egomotion_pred, [1, 2])
        # Tinghui found that scaling by a small constant facilitates training.
        egomotion_final = 0.01 * tf.reshape(
            egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return egomotion_final, end_points


def disp_net(target_image, is_training=True):
  """Predict inverse of depth from a single image."""
  batch_norm_params = {'is_training': is_training}
  h = target_image.get_shape()[1].value
  w = target_image.get_shape()[2].value
  inputs = target_image
  with tf.variable_scope('depth_net') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    normalizer_fn = slim.batch_norm if FLAGS.use_bn else None
    normalizer_params = batch_norm_params if FLAGS.use_bn else None
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=slim.l2_regularizer(WEIGHT_REG),
                        activation_fn=tf.nn.relu,
                        outputs_collections=end_points_collection):
      cnv1 = slim.conv2d(inputs, 32, [7, 7], stride=2, scope='cnv1')
      cnv1b = slim.conv2d(cnv1, 32, [7, 7], stride=1, scope='cnv1b')
      cnv2 = slim.conv2d(cnv1b, 64, [5, 5], stride=2, scope='cnv2')
      cnv2b = slim.conv2d(cnv2, 64, [5, 5], stride=1, scope='cnv2b')

      cnv3 = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3, 128, [3, 3], stride=1, scope='cnv3b')
      cnv4 = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4, 256, [3, 3], stride=1, scope='cnv4b')
      cnv5 = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5, 512, [3, 3], stride=1, scope='cnv5b')
      cnv6 = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6, 512, [3, 3], stride=1, scope='cnv6b')
      cnv7 = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
      cnv7b = slim.conv2d(cnv7, 512, [3, 3], stride=1, scope='cnv7b')

      up7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
      # There might be dimension mismatch due to uneven down/up-sampling.
      up7 = _resize_like(up7, cnv6b)
      i7_in = tf.concat([up7, cnv6b], axis=3)
      icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

      up6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
      up6 = _resize_like(up6, cnv5b)
      i6_in = tf.concat([up6, cnv5b], axis=3)
      icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

      up5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
      up5 = _resize_like(up5, cnv4b)
      i5_in = tf.concat([up5, cnv4b], axis=3)
      icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

      up4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
      i4_in = tf.concat([up4, cnv3b], axis=3)
      icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
      disp4 = (slim.conv2d(icnv4, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                           normalizer_fn=None, scope='disp4')
               * DISP_SCALING + MIN_DISP)
      disp4_up = tf.image.resize_bilinear(disp4, [np.int(h / 4), np.int(w / 4)])

      up3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
      i3_in = tf.concat([up3, cnv2b, disp4_up], axis=3)
      icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
      disp3 = (slim.conv2d(icnv3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                           normalizer_fn=None, scope='disp3')
               * DISP_SCALING + MIN_DISP)
      disp3_up = tf.image.resize_bilinear(disp3, [np.int(h / 2), np.int(w / 2)])

      up2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
      i2_in = tf.concat([up2, cnv1b, disp3_up], axis=3)
      icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
      disp2 = (slim.conv2d(icnv2, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                           normalizer_fn=None, scope='disp2')
               * DISP_SCALING + MIN_DISP)
      disp2_up = tf.image.resize_bilinear(disp2, [h, w])

      up1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
      i1_in = tf.concat([up1, disp2_up], axis=3)
      icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
      disp1 = (slim.conv2d(icnv1, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                           normalizer_fn=None, scope='disp1')
               * DISP_SCALING + MIN_DISP)

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return [disp1, disp2, disp3, disp4], end_points


def _resize_like(inputs, ref):
  i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
  r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
  if i_h == r_h and i_w == r_w:
    return inputs
  else:
    return tf.image.resize_nearest_neighbor(inputs, [r_h.value, r_w.value])
