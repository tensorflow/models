# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
r"""Converts raw LiTS numpy data to TFRecord.

The file is forked from:
https://github.com/tensorflow/tpu/blob/master/models/official/unet3d/data_preprocess/convert_lits.py
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from scipy import ndimage
import tensorflow.google.compat.v1 as tf

flags.DEFINE_string("image_file_pattern", None,
                    "path pattern to an input image npy file.")
flags.DEFINE_string("label_file_pattern", None,
                    "path pattern to an input label npy file.")
flags.DEFINE_string("output_path", None, "path to output TFRecords.")

flags.DEFINE_boolean("crop_liver_region", True,
                     "whether to crop liver region out.")
flags.DEFINE_boolean("apply_data_aug", False,
                     "whether to apply data augmentation.")

flags.DEFINE_integer("shard_start", 0,
                     "start with volume-${shard_start}.npy.")
flags.DEFINE_integer("shard_stride", 1,
                     "this process will convert "
                     "volume-${shard_start + n * shard_stride}.npy for all n.")

flags.DEFINE_integer("output_size", 128,
                     "output, cropped size along x, y, and z.")
flags.DEFINE_integer("resize_size", 192,
                     "size along x, y, and z before cropping.")

FLAGS = flags.FLAGS


def to_1hot(label):
  per_class = []
  for classes in range(3):
    per_class.append((label == classes)[..., np.newaxis])
  label = np.concatenate(per_class, axis=-1).astype(label.dtype)
  return label


def save_to_tfrecord(image, label, idx, im_id, output_path,
                     convert_label_to_1hot):
  """Save to TFRecord."""
  if convert_label_to_1hot:
    label = to_1hot(label)

  d_feature = {}
  d_feature["image/ct_image"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[image.reshape([-1]).tobytes()]))
  d_feature["image/label"] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[label.reshape([-1]).tobytes()]))

  example = tf.train.Example(features=tf.train.Features(feature=d_feature))
  serialized = example.SerializeToString()

  result_file = os.path.join(
      output_path, "instance-{}-{}.tfrecords".format(im_id, idx))
  options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  with tf.python_io.TFRecordWriter(result_file, options=options) as w:
    w.write(serialized)


def intensity_change(im):
  """Color augmentation."""
  if np.random.rand() < 0.1:
    return im
  # Randomly scale color.
  sigma = 0.05
  truncate_rad = 0.1
  im *= np.clip(np.random.normal(1.0, sigma),
                1.0 - truncate_rad, 1.0 + truncate_rad)
  return im


def rand_crop_liver(image, label, res_s, out_s,
                    apply_data_aug, augment_times=54):
  """Crop image and label; Randomly change image intensity.

  Randomly crop image and label around liver.

  Args:
    image: 3D numpy array.
    label: 3D numpy array.
    res_s: resized size of image and label.
    out_s: output size of random crops.
    apply_data_aug: whether to apply data augmentation.
    augment_times: the number of times to randomly crop and augment data.
  Yields:
    croped and augmented image and label.
  """
  if image.shape != (res_s, res_s, res_s) or \
      label.shape != (res_s, res_s, res_s):
    logging.info("Unexpected shapes. "
                 "image.shape: %s, label.shape: %s",
                 image.shape, label.shape)
    return

  rough_liver_label = 1
  x, y, z = np.where(label == rough_liver_label)
  bbox_center = [(x.min() + x.max()) // 2,
                 (y.min() + y.max()) // 2,
                 (z.min() + z.max()) // 2]

  def in_range_check(c):
    c = max(c, out_s // 2)
    c = min(c, res_s - out_s // 2)
    return c

  for _ in range(augment_times):
    rand_c = []
    for c in bbox_center:
      sigma = out_s // 6
      truncate_rad = out_s // 4
      c += np.clip(np.random.randn() * sigma, -truncate_rad, truncate_rad)
      rand_c.append(int(in_range_check(c)))

    image_aug = image[rand_c[0] - out_s // 2:rand_c[0] + out_s // 2,
                      rand_c[1] - out_s // 2:rand_c[1] + out_s // 2,
                      rand_c[2] - out_s // 2:rand_c[2] + out_s // 2].copy()
    label_aug = label[rand_c[0] - out_s // 2:rand_c[0] + out_s // 2,
                      rand_c[1] - out_s // 2:rand_c[1] + out_s // 2,
                      rand_c[2] - out_s // 2:rand_c[2] + out_s // 2].copy()

    if apply_data_aug:
      image_aug = intensity_change(image_aug)

    yield image_aug, label_aug


def rand_crop_whole_ct(image, label, res_s, out_s,
                       apply_data_aug, augment_times=2):
  """Crop image and label; Randomly change image intensity.

  Randomly crop image and label.

  Args:
    image: 3D numpy array.
    label: 3D numpy array.
    res_s: resized size of image and label.
    out_s: output size of random crops.
    apply_data_aug: whether to apply data augmentation.
    augment_times: the number of times to randomly crop and augment data.
  Yields:
    croped and augmented image and label.
  """
  if image.shape != (res_s, res_s, res_s) or \
      label.shape != (res_s, res_s, res_s):
    logging.info("Unexpected shapes. "
                 "image.shape: %s, label.shape: %s",
                 image.shape, label.shape)
    return

  if not apply_data_aug:
    # Do not augment data.
    idx = (res_s - out_s) // 2
    image = image[idx:idx + out_s, idx:idx + out_s, idx:idx + out_s]
    label = label[idx:idx + out_s, idx:idx + out_s, idx:idx + out_s]
    yield image, label
  else:
    cut = res_s - out_s
    for _ in range(augment_times):
      for i in [0, cut // 2, cut]:
        for j in [0, cut // 2, cut]:
          for k in [0, cut // 2, cut]:
            image_aug = image[i:i + out_s, j:j + out_s, k:k + out_s].copy()
            label_aug = label[i:i + out_s, j:j + out_s, k:k + out_s].copy()
            image_aug = intensity_change(image_aug)
            yield image_aug, label_aug


def resize_3d_image_nearest_interpolation(im, res_s):
  """Resize 3D image, but with nearest interpolation."""
  new_shape = [res_s, im.shape[1], im.shape[2]]
  ret0 = np.zeros(new_shape, dtype=im.dtype)
  for i in range(im.shape[2]):
    im_slice = np.array(Image.fromarray(im[..., i]).resize(
        (im.shape[1], res_s), resample=Image.NEAREST))
    ret0[..., i] = im_slice

  new_shape = [res_s, res_s, res_s]
  ret = np.zeros(new_shape, dtype=im.dtype)
  for i in range(res_s):
    im_slice = np.array(Image.fromarray(ret0[i, ...]).resize(
        (res_s, res_s), resample=Image.NEAREST))
    ret[i, ...] = im_slice
  return ret


def process_one_file(image_path, label_path, im_id,
                     output_path, res_s, out_s,
                     crop_liver_region, apply_data_aug):
  """Convert one npy file."""
  with tf.gfile.Open(image_path, "rb") as f:
    image = np.load(f)
  with tf.gfile.Open(label_path, "rb") as f:
    label = np.load(f)

  image = ndimage.zoom(image, [float(res_s) / image.shape[0],
                               float(res_s) / image.shape[1],
                               float(res_s) / image.shape[2]])
  label = resize_3d_image_nearest_interpolation(label.astype(np.uint8),
                                                res_s).astype(np.float32)

  if crop_liver_region:
    for idx, (image_aug, label_aug) in enumerate(rand_crop_liver(
        image, label, res_s, out_s, apply_data_aug)):
      save_to_tfrecord(image_aug, label_aug, idx, im_id, output_path,
                       convert_label_to_1hot=True)
  else:  # not crop_liver_region
    # If we output the entire CT scan (crop_liver_region=False),
    # do not convert_label_to_1hot to save storage.
    for idx, (image_aug, label_aug) in enumerate(rand_crop_whole_ct(
        image, label, res_s, out_s, apply_data_aug)):
      save_to_tfrecord(image_aug, label_aug, idx, im_id, output_path,
                       convert_label_to_1hot=False)


def main(argv):
  del argv

  output_path = FLAGS.output_path
  res_s = FLAGS.resize_size
  out_s = FLAGS.output_size
  crop_liver_region = FLAGS.crop_liver_region
  apply_data_aug = FLAGS.apply_data_aug

  for im_id in range(FLAGS.shard_start, 1000000, FLAGS.shard_stride):
    image_path = FLAGS.image_file_pattern.format(im_id)
    label_path = FLAGS.label_file_pattern.format(im_id)
    if not tf.gfile.Exists(image_path):
      logging.info("Reached the end. Image does not exist: %s. "
                   "Process finish.", image_path)
      break
    process_one_file(image_path, label_path, im_id,
                     output_path, res_s, out_s,
                     crop_liver_region, apply_data_aug)


if __name__ == "__main__":
  app.run(main)
