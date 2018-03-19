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
"""Wrapper for providing semantic segmentation data."""

import tensorflow as tf
from deeplab import common
from deeplab import input_preprocess

slim = tf.contrib.slim

dataset_data_provider = slim.dataset_data_provider


def _get_data(data_provider, dataset_split):
  """Gets data from data provider.

  Args:
    data_provider: An object of slim.data_provider.
    dataset_split: Dataset split.

  Returns:
    image: Image Tensor.
    label: Label Tensor storing segmentation annotations.
    image_name: Image name.
    height: Image height.
    width: Image width.

  Raises:
    ValueError: Failed to find label.
  """
  if common.LABELS_CLASS not in data_provider.list_items():
    raise ValueError('Failed to find labels.')

  image, height, width = data_provider.get(
      [common.IMAGE, common.HEIGHT, common.WIDTH])

  # Some datasets do not contain image_name.
  if common.IMAGE_NAME in data_provider.list_items():
    image_name, = data_provider.get([common.IMAGE_NAME])
  else:
    image_name = tf.constant('')

  label = None
  if dataset_split != common.TEST_SET:
    label, = data_provider.get([common.LABELS_CLASS])

  return image, label, image_name, height, width


def get(dataset,
        crop_size,
        batch_size,
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None):
  """Gets the dataset split for semantic segmentation.

  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.

  Args:
    dataset: An instance of slim Dataset.
    crop_size: Image crop size [height, width].
    batch_size: Batch size.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    dataset_split: Dataset split.
    is_training: Is training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    A dictionary of batched Tensors for semantic segmentation.

  Raises:
    ValueError: dataset_split is None, failed to find labels, or label shape
      is not valid.
  """
  if dataset_split is None:
    raise ValueError('Unknown dataset split.')
  if model_variant is None:
    tf.logging.warning('Please specify a model_variant. See '
                       'feature_extractor.network_map for supported model '
                       'variants.')

  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
  image, label, image_name, height, width = _get_data(data_provider,
                                                      dataset_split)
  if label is not None:
    if label.shape.ndims == 2:
      label = tf.expand_dims(label, 2)
    elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
      pass
    else:
      raise ValueError('Input label shape must be [height, width], or '
                       '[height, width, 1].')

    label.set_shape([None, None, 1])
  original_image, image, label = input_preprocess.preprocess_image_and_label(
      image,
      label,
      crop_height=crop_size[0],
      crop_width=crop_size[1],
      min_resize_value=min_resize_value,
      max_resize_value=max_resize_value,
      resize_factor=resize_factor,
      min_scale_factor=min_scale_factor,
      max_scale_factor=max_scale_factor,
      scale_factor_step_size=scale_factor_step_size,
      ignore_label=dataset.ignore_label,
      is_training=is_training,
      model_variant=model_variant)
  sample = {
      common.IMAGE: image,
      common.IMAGE_NAME: image_name,
      common.HEIGHT: height,
      common.WIDTH: width
  }
  if label is not None:
    sample[common.LABEL] = label

  if not is_training:
    # Original image is only used during visualization.
    sample[common.ORIGINAL_IMAGE] = original_image,
    num_threads = 1

  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)
