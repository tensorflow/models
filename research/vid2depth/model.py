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

"""Build model for inference or training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import nets
from ops import icp_grad  # pylint: disable=unused-import
from ops.icp_op import icp
import project
import reader
import tensorflow as tf
import util

gfile = tf.gfile
slim = tf.contrib.slim

NUM_SCALES = 4


class Model(object):
  """Model code from SfMLearner."""

  def __init__(self,
               data_dir=None,
               is_training=True,
               learning_rate=0.0002,
               beta1=0.9,
               reconstr_weight=0.85,
               smooth_weight=0.05,
               ssim_weight=0.15,
               icp_weight=0.0,
               batch_size=4,
               img_height=128,
               img_width=416,
               seq_length=3,
               legacy_mode=False):
    self.data_dir = data_dir
    self.is_training = is_training
    self.learning_rate = learning_rate
    self.reconstr_weight = reconstr_weight
    self.smooth_weight = smooth_weight
    self.ssim_weight = ssim_weight
    self.icp_weight = icp_weight
    self.beta1 = beta1
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.legacy_mode = legacy_mode

    logging.info('data_dir: %s', data_dir)
    logging.info('learning_rate: %s', learning_rate)
    logging.info('beta1: %s', beta1)
    logging.info('smooth_weight: %s', smooth_weight)
    logging.info('ssim_weight: %s', ssim_weight)
    logging.info('icp_weight: %s', icp_weight)
    logging.info('batch_size: %s', batch_size)
    logging.info('img_height: %s', img_height)
    logging.info('img_width: %s', img_width)
    logging.info('seq_length: %s', seq_length)
    logging.info('legacy_mode: %s', legacy_mode)

    if self.is_training:
      self.reader = reader.DataReader(self.data_dir, self.batch_size,
                                      self.img_height, self.img_width,
                                      self.seq_length, NUM_SCALES)
      self.build_train_graph()
    else:
      self.build_depth_test_graph()
      self.build_egomotion_test_graph()

    # At this point, the model is ready.  Print some info on model params.
    util.count_parameters()

  def build_train_graph(self):
    self.build_inference_for_training()
    self.build_loss()
    self.build_train_op()
    self.build_summaries()

  def build_inference_for_training(self):
    """Invokes depth and ego-motion networks and computes clouds if needed."""
    (self.image_stack, self.intrinsic_mat, self.intrinsic_mat_inv) = (
        self.reader.read_data())
    with tf.name_scope('egomotion_prediction'):
      self.egomotion, _ = nets.egomotion_net(self.image_stack, is_training=True,
                                             legacy_mode=self.legacy_mode)
    with tf.variable_scope('depth_prediction'):
      # Organized by ...[i][scale].  Note that the order is flipped in
      # variables in build_loss() below.
      self.disp = {}
      self.depth = {}
      if self.icp_weight > 0:
        self.cloud = {}
      for i in range(self.seq_length):
        image = self.image_stack[:, :, :, 3 * i:3 * (i + 1)]
        multiscale_disps_i, _ = nets.disp_net(image, is_training=True)
        multiscale_depths_i = [1.0 / d for d in multiscale_disps_i]
        self.disp[i] = multiscale_disps_i
        self.depth[i] = multiscale_depths_i
        if self.icp_weight > 0:
          multiscale_clouds_i = [
              project.get_cloud(d,
                                self.intrinsic_mat_inv[:, s, :, :],
                                name='cloud%d_%d' % (s, i))
              for (s, d) in enumerate(multiscale_depths_i)
          ]
          self.cloud[i] = multiscale_clouds_i
        # Reuse the same depth graph for all images.
        tf.get_variable_scope().reuse_variables()
    logging.info('disp: %s', util.info(self.disp))

  def build_loss(self):
    """Adds ops for computing loss."""
    with tf.name_scope('compute_loss'):
      self.reconstr_loss = 0
      self.smooth_loss = 0
      self.ssim_loss = 0
      self.icp_transform_loss = 0
      self.icp_residual_loss = 0

      # self.images is organized by ...[scale][B, h, w, seq_len * 3].
      self.images = [{} for _ in range(NUM_SCALES)]
      # Following nested lists are organized by ...[scale][source-target].
      self.warped_image = [{} for _ in range(NUM_SCALES)]
      self.warp_mask = [{} for _ in range(NUM_SCALES)]
      self.warp_error = [{} for _ in range(NUM_SCALES)]
      self.ssim_error = [{} for _ in range(NUM_SCALES)]
      self.icp_transform = [{} for _ in range(NUM_SCALES)]
      self.icp_residual = [{} for _ in range(NUM_SCALES)]

      self.middle_frame_index = util.get_seq_middle(self.seq_length)

      # Compute losses at each scale.
      for s in range(NUM_SCALES):
        # Scale image stack.
        height_s = int(self.img_height / (2**s))
        width_s = int(self.img_width / (2**s))
        self.images[s] = tf.image.resize_area(self.image_stack,
                                              [height_s, width_s])

        # Smoothness.
        if self.smooth_weight > 0:
          for i in range(self.seq_length):
            # In legacy mode, use the depth map from the middle frame only.
            if not self.legacy_mode or i == self.middle_frame_index:
              self.smooth_loss += 1.0 / (2**s) * self.depth_smoothness(
                  self.disp[i][s], self.images[s][:, :, :, 3 * i:3 * (i + 1)])

        for i in range(self.seq_length):
          for j in range(self.seq_length):
            # Only consider adjacent frames.
            if i == j or abs(i - j) != 1:
              continue
            # In legacy mode, only consider the middle frame as target.
            if self.legacy_mode and j != self.middle_frame_index:
              continue
            source = self.images[s][:, :, :, 3 * i:3 * (i + 1)]
            target = self.images[s][:, :, :, 3 * j:3 * (j + 1)]
            target_depth = self.depth[j][s]
            key = '%d-%d' % (i, j)

            # Extract ego-motion from i to j
            egomotion_index = min(i, j)
            egomotion_mult = 1
            if i > j:
              # Need to inverse egomotion when going back in sequence.
              egomotion_mult *= -1
            # For compatiblity with SfMLearner, interpret all egomotion vectors
            # as pointing toward the middle frame.  Note that unlike SfMLearner,
            # each vector captures the motion to/from its next frame, and not
            # the center frame.  Although with seq_length == 3, there is no
            # difference.
            if self.legacy_mode:
              if egomotion_index >= self.middle_frame_index:
                egomotion_mult *= -1
            egomotion = egomotion_mult * self.egomotion[:, egomotion_index, :]

            # Inverse warp the source image to the target image frame for
            # photometric consistency loss.
            self.warped_image[s][key], self.warp_mask[s][key] = (
                project.inverse_warp(source,
                                     target_depth,
                                     egomotion,
                                     self.intrinsic_mat[:, s, :, :],
                                     self.intrinsic_mat_inv[:, s, :, :]))

            # Reconstruction loss.
            self.warp_error[s][key] = tf.abs(self.warped_image[s][key] - target)
            self.reconstr_loss += tf.reduce_mean(
                self.warp_error[s][key] * self.warp_mask[s][key])
            # SSIM.
            if self.ssim_weight > 0:
              self.ssim_error[s][key] = self.ssim(self.warped_image[s][key],
                                                  target)
              # TODO(rezama): This should be min_pool2d().
              ssim_mask = slim.avg_pool2d(self.warp_mask[s][key], 3, 1, 'VALID')
              self.ssim_loss += tf.reduce_mean(
                  self.ssim_error[s][key] * ssim_mask)
            # 3D loss.
            if self.icp_weight > 0:
              cloud_a = self.cloud[j][s]
              cloud_b = self.cloud[i][s]
              self.icp_transform[s][key], self.icp_residual[s][key] = icp(
                  cloud_a, egomotion, cloud_b)
              self.icp_transform_loss += 1.0 / (2**s) * tf.reduce_mean(
                  tf.abs(self.icp_transform[s][key]))
              self.icp_residual_loss += 1.0 / (2**s) * tf.reduce_mean(
                  tf.abs(self.icp_residual[s][key]))

      self.total_loss = self.reconstr_weight * self.reconstr_loss
      if self.smooth_weight > 0:
        self.total_loss += self.smooth_weight * self.smooth_loss
      if self.ssim_weight > 0:
        self.total_loss += self.ssim_weight * self.ssim_loss
      if self.icp_weight > 0:
        self.total_loss += self.icp_weight * (self.icp_transform_loss +
                                              self.icp_residual_loss)

  def gradient_x(self, img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

  def gradient_y(self, img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

  def depth_smoothness(self, depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = self.gradient_x(depth)
    depth_dy = self.gradient_y(depth)
    image_dx = self.gradient_x(img)
    image_dy = self.gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

  def ssim(self, x, y):
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
    sigma_x = slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
    sigma_y = slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)

  def build_train_op(self):
    with tf.name_scope('train_op'):
      optim = tf.train.AdamOptimizer(self.learning_rate, self.beta1)
      self.train_op = slim.learning.create_train_op(self.total_loss, optim)
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

  def build_summaries(self):
    """Adds scalar and image summaries for TensorBoard."""
    tf.summary.scalar('total_loss', self.total_loss)
    tf.summary.scalar('reconstr_loss', self.reconstr_loss)
    if self.smooth_weight > 0:
      tf.summary.scalar('smooth_loss', self.smooth_loss)
    if self.ssim_weight > 0:
      tf.summary.scalar('ssim_loss', self.ssim_loss)
    if self.icp_weight > 0:
      tf.summary.scalar('icp_transform_loss', self.icp_transform_loss)
      tf.summary.scalar('icp_residual_loss', self.icp_residual_loss)

    for i in range(self.seq_length - 1):
      tf.summary.histogram('tx%d' % i, self.egomotion[:, i, 0])
      tf.summary.histogram('ty%d' % i, self.egomotion[:, i, 1])
      tf.summary.histogram('tz%d' % i, self.egomotion[:, i, 2])
      tf.summary.histogram('rx%d' % i, self.egomotion[:, i, 3])
      tf.summary.histogram('ry%d' % i, self.egomotion[:, i, 4])
      tf.summary.histogram('rz%d' % i, self.egomotion[:, i, 5])

    for s in range(NUM_SCALES):
      for i in range(self.seq_length):
        tf.summary.image('scale%d_image%d' % (s, i),
                         self.images[s][:, :, :, 3 * i:3 * (i + 1)])
        if i in self.depth:
          tf.summary.histogram('scale%d_depth%d' % (s, i), self.depth[i][s])
          tf.summary.histogram('scale%d_disp%d' % (s, i), self.disp[i][s])
          tf.summary.image('scale%d_disparity%d' % (s, i), self.disp[i][s])

      for key in self.warped_image[s]:
        tf.summary.image('scale%d_warped_image%s' % (s, key),
                         self.warped_image[s][key])
        tf.summary.image('scale%d_warp_mask%s' % (s, key),
                         self.warp_mask[s][key])
        tf.summary.image('scale%d_warp_error%s' % (s, key),
                         self.warp_error[s][key])
        if self.ssim_weight > 0:
          tf.summary.image('scale%d_ssim_error%s' % (s, key),
                           self.ssim_error[s][key])
        if self.icp_weight > 0:
          tf.summary.image('scale%d_icp_residual%s' % (s, key),
                           self.icp_residual[s][key])
          transform = self.icp_transform[s][key]
          tf.summary.histogram('scale%d_icp_tx%s' % (s, key), transform[:, 0])
          tf.summary.histogram('scale%d_icp_ty%s' % (s, key), transform[:, 1])
          tf.summary.histogram('scale%d_icp_tz%s' % (s, key), transform[:, 2])
          tf.summary.histogram('scale%d_icp_rx%s' % (s, key), transform[:, 3])
          tf.summary.histogram('scale%d_icp_ry%s' % (s, key), transform[:, 4])
          tf.summary.histogram('scale%d_icp_rz%s' % (s, key), transform[:, 5])

  def build_depth_test_graph(self):
    """Builds depth model reading from placeholders."""
    with tf.name_scope('depth_prediction'):
      with tf.variable_scope('depth_prediction'):
        input_uint8 = tf.placeholder(
            tf.uint8, [self.batch_size, self.img_height, self.img_width, 3],
            name='raw_input')
        input_float = tf.image.convert_image_dtype(input_uint8, tf.float32)
        # TODO(rezama): Retrain published model with batchnorm params and set
        # is_training to False.
        est_disp, _ = nets.disp_net(input_float, is_training=True)
        est_depth = 1.0 / est_disp[0]
    self.inputs_depth = input_uint8
    self.est_depth = est_depth

  def build_egomotion_test_graph(self):
    """Builds egomotion model reading from placeholders."""
    input_uint8 = tf.placeholder(
        tf.uint8,
        [self.batch_size, self.img_height, self.img_width * self.seq_length, 3],
        name='raw_input')
    input_float = tf.image.convert_image_dtype(input_uint8, tf.float32)
    image_seq = input_float
    image_stack = self.unpack_image_batches(image_seq)
    with tf.name_scope('egomotion_prediction'):
        # TODO(rezama): Retrain published model with batchnorm params and set
        # is_training to False.
      egomotion, _ = nets.egomotion_net(image_stack, is_training=True,
                                        legacy_mode=self.legacy_mode)
    self.inputs_egomotion = input_uint8
    self.est_egomotion = egomotion

  def unpack_image_batches(self, image_seq):
    """[B, h, w * seq_length, 3] -> [B, h, w, 3 * seq_length]."""
    with tf.name_scope('unpack_images'):
      image_list = [
          image_seq[:, :, i * self.img_width:(i + 1) * self.img_width, :]
          for i in range(self.seq_length)
      ]
      image_stack = tf.concat(image_list, axis=3)
      image_stack.set_shape([
          self.batch_size, self.img_height, self.img_width, self.seq_length * 3
      ])
    return image_stack

  def inference(self, inputs, sess, mode):
    """Runs depth or egomotion inference from placeholders."""
    fetches = {}
    if mode == 'depth':
      fetches['depth'] = self.est_depth
      inputs_ph = self.inputs_depth
    if mode == 'egomotion':
      fetches['egomotion'] = self.est_egomotion
      inputs_ph = self.inputs_egomotion
    results = sess.run(fetches, feed_dict={inputs_ph: inputs})
    return results
