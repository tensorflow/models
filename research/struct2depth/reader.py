
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

"""Reads data that is produced by dataset/gen_data.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from absl import logging
import tensorflow as tf

import util

gfile = tf.gfile

QUEUE_SIZE = 2000
QUEUE_BUFFER = 3
# See nets.encoder_resnet as reference for below input-normalizing constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_SD = (0.229, 0.224, 0.225)
FLIP_RANDOM = 'random'  # Always perform random flipping.
FLIP_ALWAYS = 'always'  # Always flip image input, used for test augmentation.
FLIP_NONE = 'none'  # Always disables flipping.


class DataReader(object):
  """Reads stored sequences which are produced by dataset/gen_data.py."""

  def __init__(self, data_dir, batch_size, img_height, img_width, seq_length,
               num_scales, file_extension, random_scale_crop, flipping_mode,
               random_color, imagenet_norm, shuffle, input_file='train'):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.num_scales = num_scales
    self.file_extension = file_extension
    self.random_scale_crop = random_scale_crop
    self.flipping_mode = flipping_mode
    self.random_color = random_color
    self.imagenet_norm = imagenet_norm
    self.shuffle = shuffle
    self.input_file = input_file

  def read_data(self):
    """Provides images and camera intrinsics."""
    with tf.name_scope('data_loading'):
      with tf.name_scope('enqueue_paths'):
        seed = random.randint(0, 2**31 - 1)
        self.file_lists = self.compile_file_list(self.data_dir, self.input_file)
        image_paths_queue = tf.train.string_input_producer(
            self.file_lists['image_file_list'], seed=seed,
            shuffle=self.shuffle,
            num_epochs=(1 if not self.shuffle else None)
        )
        seg_paths_queue = tf.train.string_input_producer(
            self.file_lists['segment_file_list'], seed=seed,
            shuffle=self.shuffle,
            num_epochs=(1 if not self.shuffle else None))
        cam_paths_queue = tf.train.string_input_producer(
            self.file_lists['cam_file_list'], seed=seed,
            shuffle=self.shuffle,
            num_epochs=(1 if not self.shuffle else None))
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        seg_reader = tf.WholeFileReader()
        _, seg_contents = seg_reader.read(seg_paths_queue)
        if self.file_extension == 'jpg':
          image_seq = tf.image.decode_jpeg(image_contents)
          seg_seq = tf.image.decode_jpeg(seg_contents, channels=3)
        elif self.file_extension == 'png':
          image_seq = tf.image.decode_png(image_contents, channels=3)
          seg_seq = tf.image.decode_png(seg_contents, channels=3)

      with tf.name_scope('load_intrinsics'):
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for _ in range(9):
          rec_def.append([1.0])
        raw_cam_vec = tf.decode_csv(raw_cam_contents, record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])

      with tf.name_scope('convert_image'):
        image_seq = self.preprocess_image(image_seq)  # Converts to float.

      if self.random_color:
        with tf.name_scope('image_augmentation'):
          image_seq = self.augment_image_colorspace(image_seq)

      image_stack = self.unpack_images(image_seq)
      seg_stack = self.unpack_images(seg_seq)

      if self.flipping_mode != FLIP_NONE:
        random_flipping = (self.flipping_mode == FLIP_RANDOM)
        with tf.name_scope('image_augmentation_flip'):
          image_stack, seg_stack, intrinsics = self.augment_images_flip(
              image_stack, seg_stack, intrinsics,
              randomized=random_flipping)

      if self.random_scale_crop:
        with tf.name_scope('image_augmentation_scale_crop'):
          image_stack, seg_stack, intrinsics = self.augment_images_scale_crop(
              image_stack, seg_stack, intrinsics, self.img_height,
              self.img_width)

      with tf.name_scope('multi_scale_intrinsics'):
        intrinsic_mat = self.get_multi_scale_intrinsics(intrinsics,
                                                        self.num_scales)
        intrinsic_mat.set_shape([self.num_scales, 3, 3])
        intrinsic_mat_inv = tf.matrix_inverse(intrinsic_mat)
        intrinsic_mat_inv.set_shape([self.num_scales, 3, 3])

      if self.imagenet_norm:
        im_mean = tf.tile(
            tf.constant(IMAGENET_MEAN), multiples=[self.seq_length])
        im_sd = tf.tile(
            tf.constant(IMAGENET_SD), multiples=[self.seq_length])
        image_stack_norm = (image_stack - im_mean) / im_sd
      else:
        image_stack_norm = image_stack

      with tf.name_scope('batching'):
        if self.shuffle:
          (image_stack, image_stack_norm, seg_stack, intrinsic_mat,
           intrinsic_mat_inv) = tf.train.shuffle_batch(
               [image_stack, image_stack_norm, seg_stack, intrinsic_mat,
                intrinsic_mat_inv],
               batch_size=self.batch_size,
               capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size,
               min_after_dequeue=QUEUE_SIZE)
        else:
          (image_stack, image_stack_norm, seg_stack, intrinsic_mat,
           intrinsic_mat_inv) = tf.train.batch(
               [image_stack, image_stack_norm, seg_stack, intrinsic_mat,
                intrinsic_mat_inv],
               batch_size=self.batch_size,
               num_threads=1,
               capacity=QUEUE_SIZE + QUEUE_BUFFER * self.batch_size)
        logging.info('image_stack: %s', util.info(image_stack))
    return (image_stack, image_stack_norm, seg_stack, intrinsic_mat,
            intrinsic_mat_inv)

  def unpack_images(self, image_seq):
    """[h, w * seq_length, 3] -> [h, w, 3 * seq_length]."""
    with tf.name_scope('unpack_images'):
      image_list = [
          image_seq[:, i * self.img_width:(i + 1) * self.img_width, :]
          for i in range(self.seq_length)
      ]
      image_stack = tf.concat(image_list, axis=2)
      image_stack.set_shape(
          [self.img_height, self.img_width, self.seq_length * 3])
    return image_stack

  @classmethod
  def preprocess_image(cls, image):
    # Convert from uint8 to float.
    return tf.image.convert_image_dtype(image, dtype=tf.float32)

  @classmethod
  def augment_image_colorspace(cls, image_stack):
    """Apply data augmentation to inputs."""
    image_stack_aug = image_stack
    # Randomly shift brightness.
    apply_brightness = tf.less(tf.random_uniform(
        shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    image_stack_aug = tf.cond(
        apply_brightness,
        lambda: tf.image.random_brightness(image_stack_aug, max_delta=0.1),
        lambda: image_stack_aug)

    # Randomly shift contrast.
    apply_contrast = tf.less(tf.random_uniform(
        shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    image_stack_aug = tf.cond(
        apply_contrast,
        lambda: tf.image.random_contrast(image_stack_aug, 0.85, 1.15),
        lambda: image_stack_aug)

    # Randomly change saturation.
    apply_saturation = tf.less(tf.random_uniform(
        shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    image_stack_aug = tf.cond(
        apply_saturation,
        lambda: tf.image.random_saturation(image_stack_aug, 0.85, 1.15),
        lambda: image_stack_aug)

    # Randomly change hue.
    apply_hue = tf.less(tf.random_uniform(
        shape=[], minval=0.0, maxval=1.0, dtype=tf.float32), 0.5)
    image_stack_aug = tf.cond(
        apply_hue,
        lambda: tf.image.random_hue(image_stack_aug, max_delta=0.1),
        lambda: image_stack_aug)

    image_stack_aug = tf.clip_by_value(image_stack_aug, 0, 1)
    return image_stack_aug

  @classmethod
  def augment_images_flip(cls, image_stack, seg_stack, intrinsics,
                          randomized=True):
    """Randomly flips the image horizontally."""

    def flip(cls, image_stack, seg_stack, intrinsics):
      _, in_w, _ = image_stack.get_shape().as_list()
      fx = intrinsics[0, 0]
      fy = intrinsics[1, 1]
      cx = in_w - intrinsics[0, 2]
      cy = intrinsics[1, 2]
      intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
      return (tf.image.flip_left_right(image_stack),
              tf.image.flip_left_right(seg_stack), intrinsics)

    if randomized:
      prob = tf.random_uniform(shape=[], minval=0.0, maxval=1.0,
                               dtype=tf.float32)
      predicate = tf.less(prob, 0.5)
      return tf.cond(predicate,
                     lambda: flip(cls, image_stack, seg_stack, intrinsics),
                     lambda: (image_stack, seg_stack, intrinsics))
    else:
      return flip(cls, image_stack, seg_stack, intrinsics)

  @classmethod
  def augment_images_scale_crop(cls, im, seg, intrinsics, out_h, out_w):
    """Randomly scales and crops image."""

    def scale_randomly(im, seg, intrinsics):
      """Scales image and adjust intrinsics accordingly."""
      in_h, in_w, _ = im.get_shape().as_list()
      scaling = tf.random_uniform([2], 1, 1.15)
      x_scaling = scaling[0]
      y_scaling = scaling[1]
      out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
      out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
      # Add batch.
      im = tf.expand_dims(im, 0)
      im = tf.image.resize_area(im, [out_h, out_w])
      im = im[0]
      seg = tf.expand_dims(seg, 0)
      seg = tf.image.resize_area(seg, [out_h, out_w])
      seg = seg[0]
      fx = intrinsics[0, 0] * x_scaling
      fy = intrinsics[1, 1] * y_scaling
      cx = intrinsics[0, 2] * x_scaling
      cy = intrinsics[1, 2] * y_scaling
      intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
      return im, seg, intrinsics

    # Random cropping
    def crop_randomly(im, seg, intrinsics, out_h, out_w):
      """Crops image and adjust intrinsics accordingly."""
      # batch_size, in_h, in_w, _ = im.get_shape().as_list()
      in_h, in_w, _ = tf.unstack(tf.shape(im))
      offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
      offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
      im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h, out_w)
      seg = tf.image.crop_to_bounding_box(seg, offset_y, offset_x, out_h, out_w)
      fx = intrinsics[0, 0]
      fy = intrinsics[1, 1]
      cx = intrinsics[0, 2] - tf.cast(offset_x, dtype=tf.float32)
      cy = intrinsics[1, 2] - tf.cast(offset_y, dtype=tf.float32)
      intrinsics = cls.make_intrinsics_matrix(fx, fy, cx, cy)
      return im, seg, intrinsics

    im, seg, intrinsics = scale_randomly(im, seg, intrinsics)
    im, seg, intrinsics = crop_randomly(im, seg, intrinsics, out_h, out_w)
    return im, seg, intrinsics

  def compile_file_list(self, data_dir, split, load_pose=False):
    """Creates a list of input files."""
    logging.info('data_dir: %s', data_dir)
    with gfile.Open(os.path.join(data_dir, '%s.txt' % split), 'r') as f:
      frames = f.readlines()
      frames = [k.rstrip() for k in frames]
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1] for x in frames]
    image_file_list = [
        os.path.join(data_dir, subfolders[i], frame_ids[i] + '.' +
                     self.file_extension)
        for i in range(len(frames))
    ]
    segment_file_list = [
        os.path.join(data_dir, subfolders[i], frame_ids[i] + '-fseg.' +
                     self.file_extension)
        for i in range(len(frames))
    ]
    cam_file_list = [
        os.path.join(data_dir, subfolders[i], frame_ids[i] + '_cam.txt')
        for i in range(len(frames))
    ]
    file_lists = {}
    file_lists['image_file_list'] = image_file_list
    file_lists['segment_file_list'] = segment_file_list
    file_lists['cam_file_list'] = cam_file_list
    if load_pose:
      pose_file_list = [
          os.path.join(data_dir, subfolders[i], frame_ids[i] + '_pose.txt')
          for i in range(len(frames))
      ]
      file_lists['pose_file_list'] = pose_file_list
    self.steps_per_epoch = len(image_file_list) // self.batch_size
    return file_lists

  @classmethod
  def make_intrinsics_matrix(cls, fx, fy, cx, cy):
    r1 = tf.stack([fx, 0, cx])
    r2 = tf.stack([0, fy, cy])
    r3 = tf.constant([0., 0., 1.])
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics

  @classmethod
  def get_multi_scale_intrinsics(cls, intrinsics, num_scales):
    """Returns multiple intrinsic matrices for different scales."""
    intrinsics_multi_scale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
      fx = intrinsics[0, 0] / (2**s)
      fy = intrinsics[1, 1] / (2**s)
      cx = intrinsics[0, 2] / (2**s)
      cy = intrinsics[1, 2] / (2**s)
      intrinsics_multi_scale.append(cls.make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)
    return intrinsics_multi_scale
