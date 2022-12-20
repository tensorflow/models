# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Data parser and processing for segmentation datasets."""

import tensorflow as tf
from official.vision.configs import semantic_segmentation as config_lib
from official.vision.dataloaders import decoder
from official.vision.dataloaders import parser
from official.vision.dataloaders import utils
from official.vision.ops import preprocess_ops


class Decoder(decoder.Decoder):
  """A tf.Example decoder for segmentation task."""

  def __init__(self,
               image_feature=config_lib.DenseFeatureConfig(),
               additional_dense_features=None):
    self._keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value=''),
        image_feature.feature_name:
            tf.io.FixedLenFeature((), tf.string, default_value='')
    }
    if additional_dense_features:
      for feature in additional_dense_features:
        self._keys_to_features[feature.feature_name] = tf.io.FixedLenFeature(
            (), tf.string, default_value='')

  def decode(self, serialized_example):
    return tf.io.parse_single_example(serialized_example,
                                      self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               crop_size=None,
               resize_eval_groundtruth=True,
               gt_is_matting_map=False,
               groundtruth_padded_size=None,
               ignore_label=255,
               aug_rand_hflip=False,
               preserve_aspect_ratio=True,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               dtype='float32',
               image_feature=config_lib.DenseFeatureConfig(),
               additional_dense_features=None):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      crop_size: `Tensor` or `list` for [height, width] of the crop. If
        specified a training crop of size crop_size is returned. This is useful
        for cropping original images during training while evaluating on
        original image sizes.
      resize_eval_groundtruth: `bool`, if True, eval groundtruth masks are
        resized to output_size.
      gt_is_matting_map: `bool`, if True, the expected mask is in the range
        between 0 and 255. The parser will normalize the value of the mask into
        the range between 0 and 1.
      groundtruth_padded_size: `Tensor` or `list` for [height, width]. When
        resize_eval_groundtruth is set to False, the groundtruth masks are
        padded to this size.
      ignore_label: `int` the pixel with ignore label will not used for training
        and evaluation.
      aug_rand_hflip: `bool`, if True, augment training with random horizontal
        flip.
      preserve_aspect_ratio: `bool`, if True, the aspect ratio is preserved,
        otherwise, the image is resized to output_size.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
      image_feature: the config for the image input (usually RGB). Defaults to
        the config for a 3-channel image with key = `image/encoded` and ImageNet
        dataset mean/stddev.
      additional_dense_features: `list` of DenseFeatureConfig for additional
        dense features.
    """
    self._output_size = output_size
    self._crop_size = crop_size
    self._resize_eval_groundtruth = resize_eval_groundtruth
    if (not resize_eval_groundtruth) and (groundtruth_padded_size is None):
      raise ValueError('groundtruth_padded_size ([height, width]) needs to be'
                       'specified when resize_eval_groundtruth is False.')
    self._gt_is_matting_map = gt_is_matting_map
    self._groundtruth_padded_size = groundtruth_padded_size
    self._ignore_label = ignore_label
    self._preserve_aspect_ratio = preserve_aspect_ratio

    # Data augmentation.
    self._aug_rand_hflip = aug_rand_hflip
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max

    # dtype.
    self._dtype = dtype

    self._image_feature = image_feature
    self._additional_dense_features = additional_dense_features

  def _prepare_image_and_label(self, data):
    """Prepare normalized image and label."""
    height = data['image/height']
    width = data['image/width']

    label = tf.io.decode_image(
        data['image/segmentation/class/encoded'], channels=1)
    label = tf.reshape(label, (1, height, width))
    label = tf.cast(label, tf.float32)

    image = tf.io.decode_image(
        data[self._image_feature.feature_name],
        channels=self._image_feature.num_channels,
        dtype=tf.uint8)
    image = tf.reshape(image, (height, width, self._image_feature.num_channels))
    # Normalizes the image feature with mean and std values, which are divided
    # by 255 because an uint8 image are re-scaled automatically. Images other
    # than uint8 type will be wrongly normalized.
    image = preprocess_ops.normalize_image(
        image, [mean / 255.0 for mean in self._image_feature.mean],
        [stddev / 255.0 for stddev in self._image_feature.stddev])

    if self._additional_dense_features:
      input_list = [image]
      for feature_cfg in self._additional_dense_features:
        feature = tf.io.decode_image(
            data[feature_cfg.feature_name],
            channels=feature_cfg.num_channels,
            dtype=tf.uint8)
        feature = tf.reshape(feature, (height, width, feature_cfg.num_channels))
        feature = preprocess_ops.normalize_image(
            feature, [mean / 255.0 for mean in feature_cfg.mean],
            [stddev / 255.0 for stddev in feature_cfg.stddev])
        input_list.append(feature)
      concat_input = tf.concat(input_list, axis=2)
    else:
      concat_input = image

    if not self._preserve_aspect_ratio:
      label = tf.reshape(label, [data['image/height'], data['image/width'], 1])
      concat_input = tf.image.resize(
          concat_input, self._output_size, method='bilinear')
      label = tf.image.resize(label, self._output_size, method='nearest')
      label = tf.reshape(label[:, :, -1], [1] + self._output_size)

    return concat_input, label

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)

    # Normalize the label into the range of 0 and 1 for matting groundtruth.
    # Note that the input groundtruth labels must be 0 to 255, and do not
    # contain ignore_label. For gt_is_matting_map case, ignore_label is only
    # used for padding the labels.
    if self._gt_is_matting_map:
      scale = tf.constant(255.0, dtype=tf.float32)
      scale = tf.expand_dims(scale, axis=0)
      scale = tf.expand_dims(scale, axis=0)
      label = tf.cast(label, tf.float32) / scale

    if self._crop_size:

      label = tf.reshape(label, [data['image/height'], data['image/width'], 1])
      # If output_size is specified, resize image, and label to desired
      # output_size.
      if self._output_size:
        image = tf.image.resize(image, self._output_size, method='bilinear')
        label = tf.image.resize(label, self._output_size, method='nearest')

      image_mask = tf.concat([image, label], axis=2)
      image_mask_crop = tf.image.random_crop(
          image_mask, self._crop_size + [tf.shape(image_mask)[-1]])
      image = image_mask_crop[:, :, :-1]
      label = tf.reshape(image_mask_crop[:, :, -1], [1] + self._crop_size)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, _, label = preprocess_ops.random_horizontal_flip(
          image, masks=label)

    train_image_size = self._crop_size if self._crop_size else self._output_size
    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        train_image_size,
        train_image_size,
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]

    # Pad label and make sure the padded region assigned to the ignore label.
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)
    label = preprocess_ops.resize_and_crop_masks(label, image_scale,
                                                 train_image_size, offset)
    label -= 1
    label = tf.where(
        tf.equal(label, -1), self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)
    valid_mask = tf.not_equal(label, self._ignore_label)

    labels = {
        'masks': label,
        'valid_masks': valid_mask,
        'image_info': image_info,
    }

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for training and evaluation."""
    image, label = self._prepare_image_and_label(data)

    # Binarize mask if groundtruth is a matting map
    if self._gt_is_matting_map:
      label = tf.divide(tf.cast(label, dtype=tf.float32), 255.0)
      label = utils.binarize_matting_map(label)

    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image, self._output_size, self._output_size)

    if self._resize_eval_groundtruth:
      # Resizes eval masks to match input image sizes. In that case, mean IoU
      # is computed on output_size not the original size of the images.
      image_scale = image_info[2, :]
      offset = image_info[3, :]
      label = preprocess_ops.resize_and_crop_masks(label, image_scale,
                                                   self._output_size, offset)
    else:
      label = tf.image.pad_to_bounding_box(label, 0, 0,
                                           self._groundtruth_padded_size[0],
                                           self._groundtruth_padded_size[1])

    label -= 1
    label = tf.where(
        tf.equal(label, -1), self._ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)

    valid_mask = tf.not_equal(label, self._ignore_label)
    labels = {
        'masks': label,
        'valid_masks': valid_mask,
        'image_info': image_info
    }

    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels
