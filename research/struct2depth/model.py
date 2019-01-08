
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

"""Build model for inference or training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow as tf

import nets
import project
import reader
import util

gfile = tf.gfile
slim = tf.contrib.slim

NUM_SCALES = 4


class Model(object):
  """Model code based on SfMLearner."""

  def __init__(self,
               data_dir=None,
               file_extension='png',
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
               architecture=nets.RESNET,
               imagenet_norm=True,
               weight_reg=0.05,
               exhaustive_mode=False,
               random_scale_crop=False,
               flipping_mode=reader.FLIP_RANDOM,
               random_color=True,
               depth_upsampling=True,
               depth_normalization=True,
               compute_minimum_loss=True,
               use_skip=True,
               joint_encoder=True,
               build_sum=True,
               shuffle=True,
               input_file='train',
               handle_motion=False,
               equal_weighting=False,
               size_constraint_weight=0.0,
               train_global_scale_var=True):
    self.data_dir = data_dir
    self.file_extension = file_extension
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
    self.architecture = architecture
    self.imagenet_norm = imagenet_norm
    self.weight_reg = weight_reg
    self.exhaustive_mode = exhaustive_mode
    self.random_scale_crop = random_scale_crop
    self.flipping_mode = flipping_mode
    self.random_color = random_color
    self.depth_upsampling = depth_upsampling
    self.depth_normalization = depth_normalization
    self.compute_minimum_loss = compute_minimum_loss
    self.use_skip = use_skip
    self.joint_encoder = joint_encoder
    self.build_sum = build_sum
    self.shuffle = shuffle
    self.input_file = input_file
    self.handle_motion = handle_motion
    self.equal_weighting = equal_weighting
    self.size_constraint_weight = size_constraint_weight
    self.train_global_scale_var = train_global_scale_var

    logging.info('data_dir: %s', data_dir)
    logging.info('file_extension: %s', file_extension)
    logging.info('is_training: %s', is_training)
    logging.info('learning_rate: %s', learning_rate)
    logging.info('reconstr_weight: %s', reconstr_weight)
    logging.info('smooth_weight: %s', smooth_weight)
    logging.info('ssim_weight: %s', ssim_weight)
    logging.info('icp_weight: %s', icp_weight)
    logging.info('size_constraint_weight: %s', size_constraint_weight)
    logging.info('beta1: %s', beta1)
    logging.info('batch_size: %s', batch_size)
    logging.info('img_height: %s', img_height)
    logging.info('img_width: %s', img_width)
    logging.info('seq_length: %s', seq_length)
    logging.info('architecture: %s', architecture)
    logging.info('imagenet_norm: %s', imagenet_norm)
    logging.info('weight_reg: %s', weight_reg)
    logging.info('exhaustive_mode: %s', exhaustive_mode)
    logging.info('random_scale_crop: %s', random_scale_crop)
    logging.info('flipping_mode: %s', flipping_mode)
    logging.info('random_color: %s', random_color)
    logging.info('depth_upsampling: %s', depth_upsampling)
    logging.info('depth_normalization: %s', depth_normalization)
    logging.info('compute_minimum_loss: %s', compute_minimum_loss)
    logging.info('use_skip: %s', use_skip)
    logging.info('joint_encoder: %s', joint_encoder)
    logging.info('build_sum: %s', build_sum)
    logging.info('shuffle: %s', shuffle)
    logging.info('input_file: %s', input_file)
    logging.info('handle_motion: %s', handle_motion)
    logging.info('equal_weighting: %s', equal_weighting)
    logging.info('train_global_scale_var: %s', train_global_scale_var)

    if self.size_constraint_weight > 0 or not is_training:
      self.global_scale_var = tf.Variable(
          0.1, name='global_scale_var',
          trainable=self.is_training and train_global_scale_var,
          dtype=tf.float32,
          constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    if self.is_training:
      self.reader = reader.DataReader(self.data_dir, self.batch_size,
                                      self.img_height, self.img_width,
                                      self.seq_length, NUM_SCALES,
                                      self.file_extension,
                                      self.random_scale_crop,
                                      self.flipping_mode,
                                      self.random_color,
                                      self.imagenet_norm,
                                      self.shuffle,
                                      self.input_file)
      self.build_train_graph()
    else:
      self.build_depth_test_graph()
      self.build_egomotion_test_graph()
      if self.handle_motion:
        self.build_objectmotion_test_graph()

    # At this point, the model is ready. Print some info on model params.
    util.count_parameters()

  def build_train_graph(self):
    self.build_inference_for_training()
    self.build_loss()
    self.build_train_op()
    if self.build_sum:
      self.build_summaries()

  def build_inference_for_training(self):
    """Invokes depth and ego-motion networks and computes clouds if needed."""
    (self.image_stack, self.image_stack_norm, self.seg_stack,
     self.intrinsic_mat, self.intrinsic_mat_inv) = self.reader.read_data()
    with tf.variable_scope('depth_prediction'):
      # Organized by ...[i][scale].  Note that the order is flipped in
      # variables in build_loss() below.
      self.disp = {}
      self.depth = {}
      self.depth_upsampled = {}
      self.inf_loss = 0.0
      # Organized by [i].
      disp_bottlenecks = [None] * self.seq_length

      if self.icp_weight > 0:
        self.cloud = {}
      for i in range(self.seq_length):
        image = self.image_stack_norm[:, :, :, 3 * i:3 * (i + 1)]

        multiscale_disps_i, disp_bottlenecks[i] = nets.disp_net(
            self.architecture, image, self.use_skip,
            self.weight_reg, True)
        multiscale_depths_i = [1.0 / d for d in multiscale_disps_i]
        self.disp[i] = multiscale_disps_i
        self.depth[i] = multiscale_depths_i
        if self.depth_upsampling:
          self.depth_upsampled[i] = []
          # Upsample low-resolution depth maps using differentiable bilinear
          # interpolation.
          for s in range(len(multiscale_depths_i)):
            self.depth_upsampled[i].append(tf.image.resize_bilinear(
                multiscale_depths_i[s], [self.img_height, self.img_width],
                align_corners=True))

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

    if self.handle_motion:
      # Define egomotion network. This network can see the whole scene except
      # for any moving objects as indicated by the provided segmentation masks.
      # To avoid the network getting clues of motion by tracking those masks, we
      # define the segmentation masks as the union temporally.
      with tf.variable_scope('egomotion_prediction'):
        base_input = self.image_stack_norm  # (B, H, W, 9)
        seg_input = self.seg_stack  # (B, H, W, 9)
        ref_zero = tf.constant(0, dtype=tf.uint8)
        # Motion model is currently defined for three-frame sequences.
        object_mask1 = tf.equal(seg_input[:, :, :, 0], ref_zero)
        object_mask2 = tf.equal(seg_input[:, :, :, 3], ref_zero)
        object_mask3 = tf.equal(seg_input[:, :, :, 6], ref_zero)
        mask_complete = tf.expand_dims(tf.logical_and(  # (B, H, W, 1)
            tf.logical_and(object_mask1, object_mask2), object_mask3), axis=3)
        mask_complete = tf.tile(mask_complete, (1, 1, 1, 9))  # (B, H, W, 9)
        # Now mask out base_input.
        self.mask_complete = tf.to_float(mask_complete)
        self.base_input_masked = base_input * self.mask_complete
        self.egomotion = nets.egomotion_net(
            image_stack=self.base_input_masked,
            disp_bottleneck_stack=None,
            joint_encoder=False,
            seq_length=self.seq_length,
            weight_reg=self.weight_reg)

      # Define object motion network for refinement. This network only sees
      # one object at a time over the whole sequence, and tries to estimate its
      # motion. The sequence of images are the respective warped frames.

      # For each scale, contains batch_size elements of shape (N, 2, 6).
      self.object_transforms = {}
      # For each scale, contains batch_size elements of shape (N, H, W, 9).
      self.object_masks = {}
      self.object_masks_warped = {}
      # For each scale, contains batch_size elements of size N.
      self.object_ids = {}

      self.egomotions_seq = {}
      self.warped_seq = {}
      self.inputs_objectmotion_net = {}
      with tf.variable_scope('objectmotion_prediction'):
        # First, warp raw images according to overall egomotion.
        for s in range(NUM_SCALES):
          self.warped_seq[s] = []
          self.egomotions_seq[s] = []
          for source_index in range(self.seq_length):
            egomotion_mat_i_1 = project.get_transform_mat(
                self.egomotion, source_index, 1)
            warped_image_i_1, _ = (
                project.inverse_warp(
                    self.image_stack[
                        :, :, :, source_index*3:(source_index+1)*3],
                    self.depth_upsampled[1][s],
                    egomotion_mat_i_1,
                    self.intrinsic_mat[:, 0, :, :],
                    self.intrinsic_mat_inv[:, 0, :, :]))

            self.warped_seq[s].append(warped_image_i_1)
            self.egomotions_seq[s].append(egomotion_mat_i_1)

          # Second, for every object in the segmentation mask, take its mask and
          # warp it according to the egomotion estimate. Then put a threshold to
          # binarize the warped result. Use this mask to mask out background and
          # other objects, and pass the filtered image to the object motion
          # network.
          self.object_transforms[s] = []
          self.object_masks[s] = []
          self.object_ids[s] = []
          self.object_masks_warped[s] = []
          self.inputs_objectmotion_net[s] = {}

          for i in range(self.batch_size):
            seg_sequence = self.seg_stack[i]  # (H, W, 9=3*3)
            object_ids = tf.unique(tf.reshape(seg_sequence, [-1]))[0]
            self.object_ids[s].append(object_ids)
            color_stack = []
            mask_stack = []
            mask_stack_warped = []
            for j in range(self.seq_length):
              current_image = self.warped_seq[s][j][i]  # (H, W, 3)
              current_seg = seg_sequence[:, :, j * 3:(j+1) * 3]  # (H, W, 3)

              def process_obj_mask_warp(obj_id):
                """Performs warping of the individual object masks."""
                obj_mask = tf.to_float(tf.equal(current_seg, obj_id))
                # Warp obj_mask according to overall egomotion.
                obj_mask_warped, _ = (
                    project.inverse_warp(
                        tf.expand_dims(obj_mask, axis=0),
                        # Middle frame, highest scale, batch element i:
                        tf.expand_dims(self.depth_upsampled[1][s][i], axis=0),
                        # Matrix for warping j into middle frame, batch elem. i:
                        tf.expand_dims(self.egomotions_seq[s][j][i], axis=0),
                        tf.expand_dims(self.intrinsic_mat[i, 0, :, :], axis=0),
                        tf.expand_dims(self.intrinsic_mat_inv[i, 0, :, :],
                                       axis=0)))
                obj_mask_warped = tf.squeeze(obj_mask_warped)
                obj_mask_binarized = tf.greater(  # Threshold to binarize mask.
                    obj_mask_warped, tf.constant(0.5))
                return tf.to_float(obj_mask_binarized)

              def process_obj_mask(obj_id):
                """Returns the individual object masks separately."""
                return tf.to_float(tf.equal(current_seg, obj_id))
              object_masks = tf.map_fn(  # (N, H, W, 3)
                  process_obj_mask, object_ids, dtype=tf.float32)

              if self.size_constraint_weight > 0:
                # The object segmentation masks are all in object_masks.
                # We need to measure the height of every of them, and get the
                # approximate distance.

                # self.depth_upsampled of shape (seq_length, scale, B, H, W).
                depth_pred = self.depth_upsampled[j][s][i]  # (H, W)
                def get_losses(obj_mask):
                  """Get motion constraint loss."""
                  # Find height of segment.
                  coords = tf.where(tf.greater(  # Shape (num_true, 2=yx)
                      obj_mask[:, :, 0], tf.constant(0.5, dtype=tf.float32)))
                  y_max = tf.reduce_max(coords[:, 0])
                  y_min = tf.reduce_min(coords[:, 0])
                  seg_height = y_max - y_min
                  f_y = self.intrinsic_mat[i, 0, 1, 1]
                  approx_depth = ((f_y * self.global_scale_var) /
                                  tf.to_float(seg_height))
                  reference_pred = tf.boolean_mask(
                      depth_pred, tf.greater(
                          tf.reshape(obj_mask[:, :, 0],
                                     (self.img_height, self.img_width, 1)),
                          tf.constant(0.5, dtype=tf.float32)))

                  # Establish loss on approx_depth, a scalar, and
                  # reference_pred, our dense prediction. Normalize both to
                  # prevent degenerative depth shrinking.
                  global_mean_depth_pred = tf.reduce_mean(depth_pred)
                  reference_pred /= global_mean_depth_pred
                  approx_depth /= global_mean_depth_pred
                  spatial_err = tf.abs(reference_pred - approx_depth)
                  mean_spatial_err = tf.reduce_mean(spatial_err)
                  return mean_spatial_err

                losses = tf.map_fn(
                    get_losses, object_masks, dtype=tf.float32)
                self.inf_loss += tf.reduce_mean(losses)
              object_masks_warped = tf.map_fn(  # (N, H, W, 3)
                  process_obj_mask_warp, object_ids, dtype=tf.float32)
              filtered_images = tf.map_fn(
                  lambda mask: current_image * mask, object_masks_warped,
                  dtype=tf.float32)  # (N, H, W, 3)
              color_stack.append(filtered_images)
              mask_stack.append(object_masks)
              mask_stack_warped.append(object_masks_warped)

            # For this batch-element, if there are N moving objects,
            # color_stack, mask_stack and mask_stack_warped contain both
            # seq_length elements of shape (N, H, W, 3).
            # We can now concatenate them on the last axis, creating a tensor of
            # (N, H, W, 3*3 = 9), and, assuming N does not get too large so that
            # we have enough memory, pass them in a single batch to the object
            # motion network.
            mask_stack = tf.concat(mask_stack, axis=3)  # (N, H, W, 9)
            mask_stack_warped = tf.concat(mask_stack_warped, axis=3)
            color_stack = tf.concat(color_stack, axis=3)  # (N, H, W, 9)
            all_transforms = nets.objectmotion_net(
                # We cut the gradient flow here as the object motion gradient
                # should have no saying in how the egomotion network behaves.
                # One could try just stopping the gradient for egomotion, but
                # not for the depth prediction network.
                image_stack=tf.stop_gradient(color_stack),
                disp_bottleneck_stack=None,
                joint_encoder=False,  # Joint encoder not supported.
                seq_length=self.seq_length,
                weight_reg=self.weight_reg)
            # all_transforms of shape (N, 2, 6).
            self.object_transforms[s].append(all_transforms)
            self.object_masks[s].append(mask_stack)
            self.object_masks_warped[s].append(mask_stack_warped)
            self.inputs_objectmotion_net[s][i] = color_stack
            tf.get_variable_scope().reuse_variables()
    else:
      # Don't handle motion, classic model formulation.
      with tf.name_scope('egomotion_prediction'):
        if self.joint_encoder:
          # Re-arrange disp_bottleneck_stack to be of shape
          # [B, h_hid, w_hid, c_hid * seq_length]. Currently, it is a list with
          # seq_length elements, each of dimension [B, h_hid, w_hid, c_hid].
          disp_bottleneck_stack = tf.concat(disp_bottlenecks, axis=3)
        else:
          disp_bottleneck_stack = None
        self.egomotion = nets.egomotion_net(
            image_stack=self.image_stack_norm,
            disp_bottleneck_stack=disp_bottleneck_stack,
            joint_encoder=self.joint_encoder,
            seq_length=self.seq_length,
            weight_reg=self.weight_reg)

  def build_loss(self):
    """Adds ops for computing loss."""
    with tf.name_scope('compute_loss'):
      self.reconstr_loss = 0
      self.smooth_loss = 0
      self.ssim_loss = 0
      self.icp_transform_loss = 0
      self.icp_residual_loss = 0

      # self.images is organized by ...[scale][B, h, w, seq_len * 3].
      self.images = [None for _ in range(NUM_SCALES)]
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
        if s == 0:  # Just as a precaution. TF often has interpolation bugs.
          self.images[s] = self.image_stack
        else:
          height_s = int(self.img_height / (2**s))
          width_s = int(self.img_width / (2**s))
          self.images[s] = tf.image.resize_bilinear(
              self.image_stack, [height_s, width_s], align_corners=True)

        # Smoothness.
        if self.smooth_weight > 0:
          for i in range(self.seq_length):
            # When computing minimum loss, use the depth map from the middle
            # frame only.
            if not self.compute_minimum_loss or i == self.middle_frame_index:
              disp_smoothing = self.disp[i][s]
              if self.depth_normalization:
                # Perform depth normalization, dividing by the mean.
                mean_disp = tf.reduce_mean(disp_smoothing, axis=[1, 2, 3],
                                           keep_dims=True)
                disp_input = disp_smoothing / mean_disp
              else:
                disp_input = disp_smoothing
              scaling_f = (1.0 if self.equal_weighting else 1.0 / (2**s))
              self.smooth_loss += scaling_f * self.depth_smoothness(
                  disp_input, self.images[s][:, :, :, 3 * i:3 * (i + 1)])

        self.debug_all_warped_image_batches = []
        for i in range(self.seq_length):
          for j in range(self.seq_length):
            if i == j:
              continue

            # When computing minimum loss, only consider the middle frame as
            # target.
            if self.compute_minimum_loss and j != self.middle_frame_index:
              continue
            # We only consider adjacent frames, unless either
            # compute_minimum_loss is on (where the middle frame is matched with
            # all other frames) or exhaustive_mode is on (where all frames are
            # matched with each other).
            if (not self.compute_minimum_loss and not self.exhaustive_mode and
                abs(i - j) != 1):
              continue

            selected_scale = 0 if self.depth_upsampling else s
            source = self.images[selected_scale][:, :, :, 3 * i:3 * (i + 1)]
            target = self.images[selected_scale][:, :, :, 3 * j:3 * (j + 1)]

            if self.depth_upsampling:
              target_depth = self.depth_upsampled[j][s]
            else:
              target_depth = self.depth[j][s]

            key = '%d-%d' % (i, j)

            if self.handle_motion:
              # self.seg_stack of shape (B, H, W, 9).
              # target_depth corresponds to middle frame, of shape (B, H, W, 1).

              # Now incorporate the other warping results, performed according
              # to the object motion network's predictions.
              # self.object_masks batch_size elements of (N, H, W, 9).
              # self.object_masks_warped batch_size elements of (N, H, W, 9).
              # self.object_transforms batch_size elements of (N, 2, 6).
              self.all_batches = []
              for batch_s in range(self.batch_size):
                # To warp i into j, first take the base warping (this is the
                # full image i warped into j using only the egomotion estimate).
                base_warping = self.warped_seq[s][i][batch_s]
                transform_matrices_thisbatch = tf.map_fn(
                    lambda transform: project.get_transform_mat(
                        tf.expand_dims(transform, axis=0), i, j)[0],
                    self.object_transforms[0][batch_s])

                def inverse_warp_wrapper(matrix):
                  """Wrapper for inverse warping method."""
                  warp_image, _ = (
                      project.inverse_warp(
                          tf.expand_dims(base_warping, axis=0),
                          tf.expand_dims(target_depth[batch_s], axis=0),
                          tf.expand_dims(matrix, axis=0),
                          tf.expand_dims(self.intrinsic_mat[
                              batch_s, selected_scale, :, :], axis=0),
                          tf.expand_dims(self.intrinsic_mat_inv[
                              batch_s, selected_scale, :, :], axis=0)))
                  return warp_image
                warped_images_thisbatch = tf.map_fn(
                    inverse_warp_wrapper, transform_matrices_thisbatch,
                    dtype=tf.float32)
                warped_images_thisbatch = warped_images_thisbatch[:, 0, :, :, :]
                # warped_images_thisbatch is now of shape (N, H, W, 9).

                # Combine warped frames into a single one, using the object
                # masks. Result should be (1, 128, 416, 3).
                # Essentially, we here want to sum them all up, filtered by the
                # respective object masks.
                mask_base_valid_source = tf.equal(
                    self.seg_stack[batch_s, :, :, i*3:(i+1)*3],
                    tf.constant(0, dtype=tf.uint8))
                mask_base_valid_target = tf.equal(
                    self.seg_stack[batch_s, :, :, j*3:(j+1)*3],
                    tf.constant(0, dtype=tf.uint8))
                mask_valid = tf.logical_and(
                    mask_base_valid_source, mask_base_valid_target)
                self.base_warping = base_warping * tf.to_float(mask_valid)
                background = tf.expand_dims(self.base_warping, axis=0)
                def construct_const_filter_tensor(obj_id):
                  return tf.fill(
                      dims=[self.img_height, self.img_width, 3],
                      value=tf.sign(obj_id)) * tf.to_float(
                          tf.equal(self.seg_stack[batch_s, :, :, 3:6],
                                   tf.cast(obj_id, dtype=tf.uint8)))
                filter_tensor = tf.map_fn(
                    construct_const_filter_tensor,
                    tf.to_float(self.object_ids[s][batch_s]))
                filter_tensor = tf.stack(filter_tensor, axis=0)
                objects_to_add = tf.reduce_sum(
                    tf.multiply(warped_images_thisbatch, filter_tensor),
                    axis=0, keepdims=True)
                combined = background + objects_to_add
                self.all_batches.append(combined)
               # Now of shape (B, 128, 416, 3).
              self.warped_image[s][key] = tf.concat(self.all_batches, axis=0)

            else:
              # Don't handle motion, classic model formulation.
              egomotion_mat_i_j = project.get_transform_mat(
                  self.egomotion, i, j)
              # Inverse warp the source image to the target image frame for
              # photometric consistency loss.
              self.warped_image[s][key], self.warp_mask[s][key] = (
                  project.inverse_warp(
                      source,
                      target_depth,
                      egomotion_mat_i_j,
                      self.intrinsic_mat[:, selected_scale, :, :],
                      self.intrinsic_mat_inv[:, selected_scale, :, :]))

            # Reconstruction loss.
            self.warp_error[s][key] = tf.abs(self.warped_image[s][key] - target)
            if not self.compute_minimum_loss:
              self.reconstr_loss += tf.reduce_mean(
                  self.warp_error[s][key] * self.warp_mask[s][key])
            # SSIM.
            if self.ssim_weight > 0:
              self.ssim_error[s][key] = self.ssim(self.warped_image[s][key],
                                                  target)
              # TODO(rezama): This should be min_pool2d().
              if not self.compute_minimum_loss:
                ssim_mask = slim.avg_pool2d(self.warp_mask[s][key], 3, 1,
                                            'VALID')
                self.ssim_loss += tf.reduce_mean(
                    self.ssim_error[s][key] * ssim_mask)

        # If the minimum loss should be computed, the loss calculation has been
        # postponed until here.
        if self.compute_minimum_loss:
          for frame_index in range(self.middle_frame_index):
            key1 = '%d-%d' % (frame_index, self.middle_frame_index)
            key2 = '%d-%d' % (self.seq_length - frame_index - 1,
                              self.middle_frame_index)
            logging.info('computing min error between %s and %s', key1, key2)
            min_error = tf.minimum(self.warp_error[s][key1],
                                   self.warp_error[s][key2])
            self.reconstr_loss += tf.reduce_mean(min_error)
            if self.ssim_weight > 0:  # Also compute the minimum SSIM loss.
              min_error_ssim = tf.minimum(self.ssim_error[s][key1],
                                          self.ssim_error[s][key2])
              self.ssim_loss += tf.reduce_mean(min_error_ssim)

      # Build the total loss as composed of L1 reconstruction, SSIM, smoothing
      # and object size constraint loss as appropriate.
      self.reconstr_loss *= self.reconstr_weight
      self.total_loss = self.reconstr_loss
      if self.smooth_weight > 0:
        self.smooth_loss *= self.smooth_weight
        self.total_loss += self.smooth_loss
      if self.ssim_weight > 0:
        self.ssim_loss *= self.ssim_weight
        self.total_loss += self.ssim_loss
      if self.size_constraint_weight > 0:
        self.inf_loss *= self.size_constraint_weight
        self.total_loss += self.inf_loss

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
    c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denominator.
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
      self.incr_global_step = tf.assign(
          self.global_step, self.global_step + 1)

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

    if self.size_constraint_weight > 0:
      tf.summary.scalar('inf_loss', self.inf_loss)
      tf.summary.histogram('global_scale_var', self.global_scale_var)

    if self.handle_motion:
      for s in range(NUM_SCALES):
        for batch_s in range(self.batch_size):
          whole_strip = tf.concat([self.warped_seq[s][0][batch_s],
                                   self.warped_seq[s][1][batch_s],
                                   self.warped_seq[s][2][batch_s]], axis=1)
          tf.summary.image('base_warp_batch%s_scale%s' % (batch_s, s),
                           tf.expand_dims(whole_strip, axis=0))

          whole_strip_input = tf.concat(
              [self.inputs_objectmotion_net[s][batch_s][:, :, :, 0:3],
               self.inputs_objectmotion_net[s][batch_s][:, :, :, 3:6],
               self.inputs_objectmotion_net[s][batch_s][:, :, :, 6:9]], axis=2)
          tf.summary.image('input_objectmotion_batch%s_scale%s' % (batch_s, s),
                           whole_strip_input)  # (B, H, 3*W, 3)

      for batch_s in range(self.batch_size):
        whole_strip = tf.concat([self.base_input_masked[batch_s, :, :, 0:3],
                                 self.base_input_masked[batch_s, :, :, 3:6],
                                 self.base_input_masked[batch_s, :, :, 6:9]],
                                axis=1)
        tf.summary.image('input_egomotion_batch%s' % batch_s,
                         tf.expand_dims(whole_strip, axis=0))

      # Show transform predictions (of all objects).
      for batch_s in range(self.batch_size):
        for i in range(self.seq_length - 1):
          # self.object_transforms contains batch_size elements of (N, 2, 6).
          tf.summary.histogram('batch%d_tx%d' % (batch_s, i),
                               self.object_transforms[0][batch_s][:, i, 0])
          tf.summary.histogram('batch%d_ty%d' % (batch_s, i),
                               self.object_transforms[0][batch_s][:, i, 1])
          tf.summary.histogram('batch%d_tz%d' % (batch_s, i),
                               self.object_transforms[0][batch_s][:, i, 2])
          tf.summary.histogram('batch%d_rx%d' % (batch_s, i),
                               self.object_transforms[0][batch_s][:, i, 3])
          tf.summary.histogram('batch%d_ry%d' % (batch_s, i),
                               self.object_transforms[0][batch_s][:, i, 4])
          tf.summary.histogram('batch%d_rz%d' % (batch_s, i),
                               self.object_transforms[0][batch_s][:, i, 5])

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
    with tf.variable_scope('depth_prediction'):
      input_image = tf.placeholder(
          tf.float32, [self.batch_size, self.img_height, self.img_width, 3],
          name='raw_input')
      if self.imagenet_norm:
        input_image = (input_image - reader.IMAGENET_MEAN) / reader.IMAGENET_SD
      est_disp, _ = nets.disp_net(architecture=self.architecture,
                                  image=input_image,
                                  use_skip=self.use_skip,
                                  weight_reg=self.weight_reg,
                                  is_training=True)
    est_depth = 1.0 / est_disp[0]
    self.input_image = input_image
    self.est_depth = est_depth

  def build_egomotion_test_graph(self):
    """Builds egomotion model reading from placeholders."""
    input_image_stack = tf.placeholder(
        tf.float32,
        [1, self.img_height, self.img_width, self.seq_length * 3],
        name='raw_input')
    input_bottleneck_stack = None

    if self.imagenet_norm:
      im_mean = tf.tile(
          tf.constant(reader.IMAGENET_MEAN), multiples=[self.seq_length])
      im_sd = tf.tile(
          tf.constant(reader.IMAGENET_SD), multiples=[self.seq_length])
      input_image_stack = (input_image_stack - im_mean) / im_sd

    if self.joint_encoder:
      # Pre-compute embeddings here.
      with tf.variable_scope('depth_prediction', reuse=True):
        input_bottleneck_stack = []
        encoder_selected = nets.encoder(self.architecture)
        for i in range(self.seq_length):
          input_image = input_image_stack[:, :, :, i * 3:(i + 1) * 3]
          tf.get_variable_scope().reuse_variables()
          embedding, _ = encoder_selected(
              target_image=input_image,
              weight_reg=self.weight_reg,
              is_training=True)
          input_bottleneck_stack.append(embedding)
        input_bottleneck_stack = tf.concat(input_bottleneck_stack, axis=3)

    with tf.variable_scope('egomotion_prediction'):
      est_egomotion = nets.egomotion_net(
          image_stack=input_image_stack,
          disp_bottleneck_stack=input_bottleneck_stack,
          joint_encoder=self.joint_encoder,
          seq_length=self.seq_length,
          weight_reg=self.weight_reg)
    self.input_image_stack = input_image_stack
    self.est_egomotion = est_egomotion

  def build_objectmotion_test_graph(self):
    """Builds egomotion model reading from placeholders."""
    input_image_stack_om = tf.placeholder(
        tf.float32,
        [1, self.img_height, self.img_width, self.seq_length * 3],
        name='raw_input')

    if self.imagenet_norm:
      im_mean = tf.tile(
          tf.constant(reader.IMAGENET_MEAN), multiples=[self.seq_length])
      im_sd = tf.tile(
          tf.constant(reader.IMAGENET_SD), multiples=[self.seq_length])
      input_image_stack_om = (input_image_stack_om - im_mean) / im_sd

    with tf.variable_scope('objectmotion_prediction'):
      est_objectmotion = nets.objectmotion_net(
          image_stack=input_image_stack_om,
          disp_bottleneck_stack=None,
          joint_encoder=self.joint_encoder,
          seq_length=self.seq_length,
          weight_reg=self.weight_reg)
    self.input_image_stack_om = input_image_stack_om
    self.est_objectmotion = est_objectmotion

  def inference_depth(self, inputs, sess):
    return sess.run(self.est_depth, feed_dict={self.input_image: inputs})

  def inference_egomotion(self, inputs, sess):
    return sess.run(
        self.est_egomotion, feed_dict={self.input_image_stack: inputs})

  def inference_objectmotion(self, inputs, sess):
    return sess.run(
        self.est_objectmotion, feed_dict={self.input_image_stack_om: inputs})
