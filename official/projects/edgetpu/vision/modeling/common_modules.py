# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Common modeling utilities."""
from typing import Optional, Tuple
# Import libraries
import numpy as np
import tensorflow as tf, tf_keras
import tensorflow.compat.v1 as tf1

from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import


MEAN_RGB = (0.5 * 255, 0.5 * 255, 0.5 * 255)
STDDEV_RGB = (0.5 * 255, 0.5 * 255, 0.5 * 255)


@tf_keras.utils.register_keras_serializable(package='Vision')
class TpuBatchNormalization(tf_keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused: Optional[bool] = False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t: tf.Tensor, num_shards_per_group: int):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError(
            'num_shards: %d mod shards_per_group: %d, should be 0' %
            (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
    return tf1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)

  def _moments(self,
               inputs: tf.Tensor,
               reduction_axes: int,
               keep_dims: int,
               mask: Optional[tf.Tensor] = None):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims, mask=mask)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 8)
    if num_shards_per_group > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
      group_mean_of_square = self._cross_replica_average(
          shard_mean_of_square, num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


def get_batch_norm(batch_norm_type: str) -> tf_keras.layers.BatchNormalization:
  """A helper to create a batch normalization getter.

  Args:
    batch_norm_type: The type of batch normalization layer implementation. `tpu`
     will use `TpuBatchNormalization`.

  Returns:
    An instance of `tf_keras.layers.BatchNormalization`.
  """
  if batch_norm_type == 'tpu':
    return TpuBatchNormalization

  return tf_keras.layers.BatchNormalization  # pytype: disable=bad-return-type  # typed-keras


def count_params(model, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return model.count_params()
  else:
    return int(np.sum([tf_keras.backend.count_params(p)
                       for p in model.trainable_weights]))


def load_weights(model: tf_keras.Model,
                 model_weights_path: str,
                 checkpoint_format: str = 'tf_checkpoint'):
  """Load model weights from the given file path.

  Args:
    model: the model to load weights into
    model_weights_path: the path of the model weights
    checkpoint_format: The source of checkpoint files. By default, we assume the
      checkpoint is saved by tf.train.Checkpoint().save(). For legacy reasons,
      we can also resotre checkpoint from keras model.save_weights() method by
      setting checkpoint_format = 'keras_checkpoint'.
  """
  if checkpoint_format == 'tf_checkpoint':
    checkpoint_dict = {'model': model}
    checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    checkpoint.restore(model_weights_path).assert_existing_objects_matched()
  elif checkpoint_format == 'keras_checkpoint':
    # Assert makes sure load is successeful.
    model.load_weights(model_weights_path).assert_existing_objects_matched()
  else:
    raise ValueError(f'Unsupported checkpoint format {checkpoint_format}.')


def normalize_images(
    features: tf.Tensor,
    num_channels: int = 3,
    dtype: str = 'float32',
    data_format: str = 'channels_last',
    mean_rgb: Tuple[float, ...] = MEAN_RGB,
    stddev_rgb: Tuple[float, ...] = STDDEV_RGB,
) -> tf.Tensor:
  """Normalizes the input image channels with the given mean and stddev.

  Args:
    features: `Tensor` representing decoded images in float format.
    num_channels: the number of channels in the input image tensor.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.
    data_format: the format of the input image tensor ['channels_first',
      'channels_last'].
    mean_rgb: the mean of the channels to subtract.
    stddev_rgb: the stddev of the channels to divide.

  Returns:
    A normalized image `Tensor`.
  """
  if data_format == 'channels_first':
    stats_shape = [num_channels, 1, 1]
  else:
    stats_shape = [1, 1, num_channels]

  if dtype is not None:
    if dtype == 'bfloat16':
      features = tf.image.convert_image_dtype(features, dtype=tf.bfloat16)

  if mean_rgb is not None:
    mean_rgb = tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
    mean_rgb = tf.broadcast_to(mean_rgb, tf.shape(features))
    features = features - mean_rgb

  if stddev_rgb is not None:
    stddev_rgb = tf.constant(
        stddev_rgb, shape=stats_shape, dtype=features.dtype)
    stddev_rgb = tf.broadcast_to(stddev_rgb, tf.shape(features))
    features = features / stddev_rgb

  return features
