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

"""Prepare the data used for FEELVOS training/evaluation."""
import tensorflow as tf

from deeplab.core import feature_extractor
from deeplab.core import preprocess_utils

# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5

get_random_scale = preprocess_utils.get_random_scale
randomly_scale_image_and_label = (
    preprocess_utils.randomly_scale_image_and_label)


def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image

  processed_image = tf.cast(image, tf.float32)

  if label is not None:
    label = tf.cast(label, tf.int32)

  # Resize image and label to the desired range.
  if min_resize_value is not None or max_resize_value is not None:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  # Data augmentation by randomly scaling the inputs.
  scale = get_random_scale(
      min_scale_factor, max_scale_factor, scale_factor_step_size)
  processed_image, label = randomly_scale_image_and_label(
      processed_image, label, scale)

  processed_image.set_shape([None, None, 3])

  if crop_height is not None and crop_width is not None:
    # Pad image and label to have dimensions >= [crop_height, crop_width].
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(
        feature_extractor.mean_pixel(model_variant), [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
      label = preprocess_utils.pad_to_bounding_box(
          label, 0, 0, target_height, target_width, ignore_label)

    # Randomly crop the image and label.
    if is_training and label is not None:
      processed_image, label = preprocess_utils.random_crop(
          [processed_image, label], crop_height, crop_width)

    processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
      label.set_shape([crop_height, crop_width, 1])

  if is_training:
    # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)

  return original_image, processed_image, label


def preprocess_images_and_labels_consistently(images,
                                              labels,
                                              crop_height,
                                              crop_width,
                                              min_resize_value=None,
                                              max_resize_value=None,
                                              resize_factor=None,
                                              min_scale_factor=1.,
                                              max_scale_factor=1.,
                                              scale_factor_step_size=0,
                                              ignore_label=255,
                                              is_training=True,
                                              model_variant=None):
  """Preprocesses images and labels in a consistent way.

  Similar to preprocess_image_and_label, but works on a list of images
  and a list of labels and uses the same crop coordinates and either flips
  all images and labels or none of them.

  Args:
    images: List of input images.
    labels: List of ground truth annotation labels.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_images: Original images (could be resized).
    processed_images: Preprocessed images.
    labels: Preprocessed ground truth segmentation labels.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and labels is None:
    raise ValueError('During training, labels must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')
  if labels is not None:
    assert len(images) == len(labels)
  num_imgs = len(images)

  # Keep reference to original images.
  original_images = images

  processed_images = [tf.cast(image, tf.float32) for image in images]

  if labels is not None:
    labels = [tf.cast(label, tf.int32) for label in labels]

  # Resize images and labels to the desired range.
  if min_resize_value is not None or max_resize_value is not None:
    processed_images, labels = zip(*[
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True) for processed_image, label
        in zip(processed_images, labels)])
    # The `original_images` becomes the resized images.
    original_images = [tf.identity(processed_image)
                       for processed_image in processed_images]

  # Data augmentation by randomly scaling the inputs.
  scale = get_random_scale(
      min_scale_factor, max_scale_factor, scale_factor_step_size)
  processed_images, labels = zip(
      *[randomly_scale_image_and_label(processed_image, label, scale)
        for processed_image, label in zip(processed_images, labels)])

  for processed_image in processed_images:
    processed_image.set_shape([None, None, 3])

  if crop_height is not None and crop_width is not None:
    # Pad image and label to have dimensions >= [crop_height, crop_width].
    image_shape = tf.shape(processed_images[0])
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

    # Pad image with mean pixel value.
    mean_pixel = tf.reshape(
        feature_extractor.mean_pixel(model_variant), [1, 1, 3])
    processed_images = [preprocess_utils.pad_to_bounding_box(
        processed_image, 0, 0, target_height, target_width, mean_pixel)
                        for processed_image in processed_images]

    if labels is not None:
      labels = [preprocess_utils.pad_to_bounding_box(
          label, 0, 0, target_height, target_width, ignore_label)
                for label in labels]

    # Randomly crop the images and labels.
    if is_training and labels is not None:
      cropped = preprocess_utils.random_crop(
          processed_images + labels, crop_height, crop_width)
      assert len(cropped) == 2 * num_imgs
      processed_images = cropped[:num_imgs]
      labels = cropped[num_imgs:]

    for processed_image in processed_images:
      processed_image.set_shape([crop_height, crop_width, 3])

    if labels is not None:
      for label in labels:
        label.set_shape([crop_height, crop_width, 1])

  if is_training:
    # Randomly left-right flip the image and label.
    res = preprocess_utils.flip_dim(
        list(processed_images + labels), _PROB_OF_FLIP, dim=1)
    maybe_flipped = res[:-1]
    assert len(maybe_flipped) == 2 * num_imgs
    processed_images = maybe_flipped[:num_imgs]
    labels = maybe_flipped[num_imgs:]

  return original_images, processed_images, labels
