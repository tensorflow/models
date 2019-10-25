
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

SIMPLE = 'simple'
RESNET = 'resnet'
ARCHITECTURES = [SIMPLE, RESNET]

SCALE_TRANSLATION = 0.001
SCALE_ROTATION = 0.01

# Disparity (inverse depth) values range from 0.01 to 10. Note that effectively,
# this is undone if depth normalization is used, which scales the values to
# have a mean of 1.
DISP_SCALING = 10
MIN_DISP = 0.01
WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'
EGOMOTION_VEC_SIZE = 6


def egomotion_net(image_stack, disp_bottleneck_stack, joint_encoder, seq_length,
                  weight_reg):
  """Predict ego-motion vectors from a stack of frames or embeddings.

  Args:
    image_stack: Input tensor with shape [B, h, w, seq_length * 3] in order.
    disp_bottleneck_stack: Input tensor with shape [B, h_hidden, w_hidden,
        seq_length * c_hidden] in order.
    joint_encoder: Determines if the same encoder is used for computing the
        bottleneck layer of both the egomotion and the depth prediction
        network. If enabled, disp_bottleneck_stack is used as input, and the
        encoding steps are skipped. If disabled, a separate encoder is defined
        on image_stack.
    seq_length: The sequence length used.
    weight_reg: The amount of weight regularization.

  Returns:
    Egomotion vectors with shape [B, seq_length - 1, 6].
  """
  num_egomotion_vecs = seq_length - 1
  with tf.variable_scope('pose_exp_net') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_reg),
                        normalizer_params=None,
                        activation_fn=tf.nn.relu,
                        outputs_collections=end_points_collection):
      if not joint_encoder:
        # Define separate encoder. If sharing, we can skip the encoding step,
        # as the bottleneck layer will already be passed as input.
        cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
        cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
        cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
        cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
        cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')

      with tf.variable_scope('pose'):
        inputs = disp_bottleneck_stack if joint_encoder else cnv5
        cnv6 = slim.conv2d(inputs, 256, [3, 3], stride=2, scope='cnv6')
        cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
        pred_channels = EGOMOTION_VEC_SIZE * num_egomotion_vecs
        egomotion_pred = slim.conv2d(cnv7, pred_channels, [1, 1], scope='pred',
                                     stride=1, normalizer_fn=None,
                                     activation_fn=None)
        egomotion_avg = tf.reduce_mean(egomotion_pred, [1, 2])
        egomotion_res = tf.reshape(
            egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])
        # Tinghui found that scaling by a small constant facilitates training.
        egomotion_scaled = tf.concat([egomotion_res[:, 0:3] * SCALE_TRANSLATION,
                                      egomotion_res[:, 3:6] * SCALE_ROTATION],
                                     axis=1)
    return egomotion_scaled


def objectmotion_net(image_stack, disp_bottleneck_stack, joint_encoder,
                     seq_length, weight_reg):
  """Predict object-motion vectors from a stack of frames or embeddings.

  Args:
    image_stack: Input tensor with shape [B, h, w, seq_length * 3] in order.
    disp_bottleneck_stack: Input tensor with shape [B, h_hidden, w_hidden,
        seq_length * c_hidden] in order.
    joint_encoder: Determines if the same encoder is used for computing the
        bottleneck layer of both the egomotion and the depth prediction
        network. If enabled, disp_bottleneck_stack is used as input, and the
        encoding steps are skipped. If disabled, a separate encoder is defined
        on image_stack.
    seq_length: The sequence length used.
    weight_reg: The amount of weight regularization.

  Returns:
    Egomotion vectors with shape [B, seq_length - 1, 6].
  """
  num_egomotion_vecs = seq_length - 1
  with tf.variable_scope('pose_exp_net') as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(weight_reg),
                        normalizer_params=None,
                        activation_fn=tf.nn.relu,
                        outputs_collections=end_points_collection):
      if not joint_encoder:
        # Define separate encoder. If sharing, we can skip the encoding step,
        # as the bottleneck layer will already be passed as input.
        cnv1 = slim.conv2d(image_stack, 16, [7, 7], stride=2, scope='cnv1')
        cnv2 = slim.conv2d(cnv1, 32, [5, 5], stride=2, scope='cnv2')
        cnv3 = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3')
        cnv4 = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
        cnv5 = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')

      with tf.variable_scope('pose'):
        inputs = disp_bottleneck_stack if joint_encoder else cnv5
        cnv6 = slim.conv2d(inputs, 256, [3, 3], stride=2, scope='cnv6')
        cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
        pred_channels = EGOMOTION_VEC_SIZE * num_egomotion_vecs
        egomotion_pred = slim.conv2d(cnv7, pred_channels, [1, 1], scope='pred',
                                     stride=1, normalizer_fn=None,
                                     activation_fn=None)
        egomotion_avg = tf.reduce_mean(egomotion_pred, [1, 2])
        egomotion_res = tf.reshape(
            egomotion_avg, [-1, num_egomotion_vecs, EGOMOTION_VEC_SIZE])
        # Tinghui found that scaling by a small constant facilitates training.
        egomotion_scaled = tf.concat([egomotion_res[:, 0:3] * SCALE_TRANSLATION,
                                      egomotion_res[:, 3:6] * SCALE_ROTATION],
                                     axis=1)
    return egomotion_scaled


def disp_net(architecture, image, use_skip, weight_reg, is_training):
  """Defines an encoder-decoder architecture for depth prediction."""
  if architecture not in ARCHITECTURES:
    raise ValueError('Unknown architecture.')
  encoder_selected = encoder(architecture)
  decoder_selected = decoder(architecture)

  # Encode image.
  bottleneck, skip_connections = encoder_selected(image, weight_reg,
                                                  is_training)
  # Decode to depth.
  multiscale_disps_i = decoder_selected(target_image=image,
                                        bottleneck=bottleneck,
                                        weight_reg=weight_reg,
                                        use_skip=use_skip,
                                        skip_connections=skip_connections)
  return multiscale_disps_i, bottleneck


def encoder(architecture):
  return encoder_resnet if architecture == RESNET else encoder_simple


def decoder(architecture):
  return decoder_resnet if architecture == RESNET else decoder_simple


def encoder_simple(target_image, weight_reg, is_training):
  """Defines the old encoding architecture."""
  del is_training
  with slim.arg_scope([slim.conv2d],
                      normalizer_fn=None,
                      normalizer_params=None,
                      weights_regularizer=slim.l2_regularizer(weight_reg),
                      activation_fn=tf.nn.relu):
    # Define (joint) encoder.
    cnv1 = slim.conv2d(target_image, 32, [7, 7], stride=2, scope='cnv1')
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
  return cnv7b, (cnv6b, cnv5b, cnv4b, cnv3b, cnv2b, cnv1b)


def decoder_simple(target_image, bottleneck, weight_reg, use_skip,
                   skip_connections):
  """Defines the old depth decoder architecture."""
  h = target_image.get_shape()[1].value
  w = target_image.get_shape()[2].value
  (cnv6b, cnv5b, cnv4b, cnv3b, cnv2b, cnv1b) = skip_connections
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      normalizer_fn=None,
                      normalizer_params=None,
                      weights_regularizer=slim.l2_regularizer(weight_reg),
                      activation_fn=tf.nn.relu):
    up7 = slim.conv2d_transpose(bottleneck, 512, [3, 3], stride=2,
                                scope='upcnv7')
    up7 = _resize_like(up7, cnv6b)
    if use_skip:
      i7_in = tf.concat([up7, cnv6b], axis=3)
    else:
      i7_in = up7
    icnv7 = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

    up6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
    up6 = _resize_like(up6, cnv5b)
    if use_skip:
      i6_in = tf.concat([up6, cnv5b], axis=3)
    else:
      i6_in = up6
    icnv6 = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

    up5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
    up5 = _resize_like(up5, cnv4b)
    if use_skip:
      i5_in = tf.concat([up5, cnv4b], axis=3)
    else:
      i5_in = up5
    icnv5 = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

    up4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
    up4 = _resize_like(up4, cnv3b)
    if use_skip:
      i4_in = tf.concat([up4, cnv3b], axis=3)
    else:
      i4_in = up4
    icnv4 = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
    disp4 = (slim.conv2d(icnv4, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                         normalizer_fn=None, scope='disp4')
             * DISP_SCALING + MIN_DISP)
    disp4_up = tf.image.resize_bilinear(disp4, [np.int(h / 4), np.int(w / 4)],
                                        align_corners=True)

    up3 = slim.conv2d_transpose(icnv4, 64, [3, 3], stride=2, scope='upcnv3')
    up3 = _resize_like(up3, cnv2b)
    if use_skip:
      i3_in = tf.concat([up3, cnv2b, disp4_up], axis=3)
    else:
      i3_in = tf.concat([up3, disp4_up])
    icnv3 = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
    disp3 = (slim.conv2d(icnv3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                         normalizer_fn=None, scope='disp3')
             * DISP_SCALING + MIN_DISP)
    disp3_up = tf.image.resize_bilinear(disp3, [np.int(h / 2), np.int(w / 2)],
                                        align_corners=True)

    up2 = slim.conv2d_transpose(icnv3, 32, [3, 3], stride=2, scope='upcnv2')
    up2 = _resize_like(up2, cnv1b)
    if use_skip:
      i2_in = tf.concat([up2, cnv1b, disp3_up], axis=3)
    else:
      i2_in = tf.concat([up2, disp3_up])
    icnv2 = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
    disp2 = (slim.conv2d(icnv2, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                         normalizer_fn=None, scope='disp2')
             * DISP_SCALING + MIN_DISP)
    disp2_up = tf.image.resize_bilinear(disp2, [h, w], align_corners=True)

    up1 = slim.conv2d_transpose(icnv2, 16, [3, 3], stride=2, scope='upcnv1')
    i1_in = tf.concat([up1, disp2_up], axis=3)
    icnv1 = slim.conv2d(i1_in, 16, [3, 3], stride=1, scope='icnv1')
    disp1 = (slim.conv2d(icnv1, 1, [3, 3], stride=1, activation_fn=tf.sigmoid,
                         normalizer_fn=None, scope='disp1')
             * DISP_SCALING + MIN_DISP)
  return [disp1, disp2, disp3, disp4]


def encoder_resnet(target_image, weight_reg, is_training):
  """Defines a ResNet18-based encoding architecture.

  This implementation follows Juyong Kim's implementation of ResNet18 on GitHub:
  https://github.com/dalgu90/resnet-18-tensorflow

  Args:
    target_image: Input tensor with shape [B, h, w, 3] to encode.
    weight_reg: Parameter ignored.
    is_training: Whether the model is being trained or not.

  Returns:
    Tuple of tensors, with the first being the bottleneck layer as tensor of
    size [B, h_hid, w_hid, c_hid], and others being intermediate layers
    for building skip-connections.
  """
  del weight_reg
  encoder_filters = [64, 64, 128, 256, 512]
  stride = 2

  # conv1
  with tf.variable_scope('conv1'):
    x = _conv(target_image, 7, encoder_filters[0], stride)
    x = _bn(x, is_train=is_training)
    econv1 = _relu(x)
    x = tf.nn.max_pool(econv1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

  # conv2_x
  x = _residual_block(x, is_training, name='conv2_1')
  econv2 = _residual_block(x, is_training, name='conv2_2')

  # conv3_x
  x = _residual_block_first(econv2, is_training, encoder_filters[2], stride,
                            name='conv3_1')
  econv3 = _residual_block(x, is_training, name='conv3_2')

  # conv4_x
  x = _residual_block_first(econv3, is_training, encoder_filters[3], stride,
                            name='conv4_1')
  econv4 = _residual_block(x, is_training, name='conv4_2')

  # conv5_x
  x = _residual_block_first(econv4, is_training, encoder_filters[4], stride,
                            name='conv5_1')
  econv5 = _residual_block(x, is_training, name='conv5_2')
  return econv5, (econv4, econv3, econv2, econv1)


def decoder_resnet(target_image, bottleneck, weight_reg, use_skip,
                   skip_connections):
  """Defines the depth decoder architecture.

  Args:
    target_image: The original encoder input tensor with shape [B, h, w, 3].
                  Just the shape information is used here.
    bottleneck: Bottleneck layer to be decoded.
    weight_reg: The amount of weight regularization.
    use_skip: Whether the passed skip connections econv1, econv2, econv3 and
              econv4 should be used.
    skip_connections: Tensors for building skip-connections.

  Returns:
    Disparities at 4 different scales.
  """
  (econv4, econv3, econv2, econv1) = skip_connections
  decoder_filters = [16, 32, 64, 128, 256]
  default_pad = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
  reg = slim.l2_regularizer(weight_reg) if weight_reg > 0.0 else None
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      normalizer_fn=None,
                      normalizer_params=None,
                      activation_fn=tf.nn.relu,
                      weights_regularizer=reg):
    upconv5 = slim.conv2d_transpose(bottleneck, decoder_filters[4], [3, 3],
                                    stride=2, scope='upconv5')
    upconv5 = _resize_like(upconv5, econv4)
    if use_skip:
      i5_in = tf.concat([upconv5, econv4], axis=3)
    else:
      i5_in = upconv5
    i5_in = tf.pad(i5_in, default_pad, mode='REFLECT')
    iconv5 = slim.conv2d(i5_in, decoder_filters[4], [3, 3], stride=1,
                         scope='iconv5', padding='VALID')

    upconv4 = slim.conv2d_transpose(iconv5, decoder_filters[3], [3, 3],
                                    stride=2, scope='upconv4')
    upconv4 = _resize_like(upconv4, econv3)
    if use_skip:
      i4_in = tf.concat([upconv4, econv3], axis=3)
    else:
      i4_in = upconv4
    i4_in = tf.pad(i4_in, default_pad, mode='REFLECT')
    iconv4 = slim.conv2d(i4_in, decoder_filters[3], [3, 3], stride=1,
                         scope='iconv4', padding='VALID')

    disp4_input = tf.pad(iconv4, default_pad, mode='REFLECT')
    disp4 = (slim.conv2d(disp4_input, 1, [3, 3], stride=1,
                         activation_fn=tf.sigmoid, normalizer_fn=None,
                         scope='disp4', padding='VALID')
             * DISP_SCALING + MIN_DISP)

    upconv3 = slim.conv2d_transpose(iconv4, decoder_filters[2], [3, 3],
                                    stride=2, scope='upconv3')
    upconv3 = _resize_like(upconv3, econv2)
    if use_skip:
      i3_in = tf.concat([upconv3, econv2], axis=3)
    else:
      i3_in = upconv3
    i3_in = tf.pad(i3_in, default_pad, mode='REFLECT')
    iconv3 = slim.conv2d(i3_in, decoder_filters[2], [3, 3], stride=1,
                         scope='iconv3', padding='VALID')
    disp3_input = tf.pad(iconv3, default_pad, mode='REFLECT')
    disp3 = (slim.conv2d(disp3_input, 1, [3, 3], stride=1,
                         activation_fn=tf.sigmoid, normalizer_fn=None,
                         scope='disp3', padding='VALID')
             * DISP_SCALING + MIN_DISP)

    upconv2 = slim.conv2d_transpose(iconv3, decoder_filters[1], [3, 3],
                                    stride=2, scope='upconv2')
    upconv2 = _resize_like(upconv2, econv1)
    if use_skip:
      i2_in = tf.concat([upconv2, econv1], axis=3)
    else:
      i2_in = upconv2
    i2_in = tf.pad(i2_in, default_pad, mode='REFLECT')
    iconv2 = slim.conv2d(i2_in, decoder_filters[1], [3, 3], stride=1,
                         scope='iconv2', padding='VALID')
    disp2_input = tf.pad(iconv2, default_pad, mode='REFLECT')
    disp2 = (slim.conv2d(disp2_input, 1, [3, 3], stride=1,
                         activation_fn=tf.sigmoid, normalizer_fn=None,
                         scope='disp2', padding='VALID')
             * DISP_SCALING + MIN_DISP)

    upconv1 = slim.conv2d_transpose(iconv2, decoder_filters[0], [3, 3],
                                    stride=2, scope='upconv1')
    upconv1 = _resize_like(upconv1, target_image)
    upconv1 = tf.pad(upconv1, default_pad, mode='REFLECT')
    iconv1 = slim.conv2d(upconv1, decoder_filters[0], [3, 3], stride=1,
                         scope='iconv1', padding='VALID')
    disp1_input = tf.pad(iconv1, default_pad, mode='REFLECT')
    disp1 = (slim.conv2d(disp1_input, 1, [3, 3], stride=1,
                         activation_fn=tf.sigmoid, normalizer_fn=None,
                         scope='disp1', padding='VALID')
             * DISP_SCALING + MIN_DISP)

  return [disp1, disp2, disp3, disp4]


def _residual_block_first(x, is_training, out_channel, strides, name='unit'):
  """Helper function for defining ResNet architecture."""
  in_channel = x.get_shape().as_list()[-1]
  with tf.variable_scope(name):
    # Shortcut connection
    if in_channel == out_channel:
      if strides == 1:
        shortcut = tf.identity(x)
      else:
        shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                  [1, strides, strides, 1], 'VALID')
    else:
      shortcut = _conv(x, 1, out_channel, strides, name='shortcut')
    # Residual
    x = _conv(x, 3, out_channel, strides, name='conv_1')
    x = _bn(x, is_train=is_training, name='bn_1')
    x = _relu(x, name='relu_1')
    x = _conv(x, 3, out_channel, 1, name='conv_2')
    x = _bn(x, is_train=is_training, name='bn_2')
    # Merge
    x = x + shortcut
    x = _relu(x, name='relu_2')
  return x


def _residual_block(x, is_training, input_q=None, output_q=None, name='unit'):
  """Helper function for defining ResNet architecture."""
  num_channel = x.get_shape().as_list()[-1]
  with tf.variable_scope(name):
    shortcut = x  # Shortcut connection
    # Residual
    x = _conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q,
              name='conv_1')
    x = _bn(x, is_train=is_training, name='bn_1')
    x = _relu(x, name='relu_1')
    x = _conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q,
              name='conv_2')
    x = _bn(x, is_train=is_training, name='bn_2')
    # Merge
    x = x + shortcut
    x = _relu(x, name='relu_2')
  return x


def _conv(x, filter_size, out_channel, stride, pad='SAME', input_q=None,
          output_q=None, name='conv'):
  """Helper function for defining ResNet architecture."""
  if (input_q is None) ^ (output_q is None):
    raise ValueError('Input/Output splits are not correctly given.')

  in_shape = x.get_shape()
  with tf.variable_scope(name):
    # Main operation: conv2d
    with tf.device('/CPU:0'):
      kernel = tf.get_variable(
          'kernel', [filter_size, filter_size, in_shape[3], out_channel],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
    if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
      tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
    conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
  return conv


def _bn(x, is_train, name='bn'):
  """Helper function for defining ResNet architecture."""
  bn = tf.layers.batch_normalization(x, training=is_train, name=name)
  return bn


def _relu(x, name=None, leakness=0.0):
  """Helper function for defining ResNet architecture."""
  if leakness > 0.0:
    name = 'lrelu' if name is None else name
    return tf.maximum(x, x*leakness, name='lrelu')
  else:
    name = 'relu' if name is None else name
    return tf.nn.relu(x, name='relu')


def _resize_like(inputs, ref):
  i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
  r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
  if i_h == r_h and i_w == r_w:
    return inputs
  else:
    # TODO(casser): Other interpolation methods could be explored here.
    return tf.image.resize_bilinear(inputs, [r_h.value, r_w.value],
                                    align_corners=True)
