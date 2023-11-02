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

"""Contains definitions of ROI aligner."""

from typing import Mapping
import tensorflow as tf, tf_keras

from official.vision.ops import spatial_transform_ops


@tf_keras.utils.register_keras_serializable(package='Vision')
class MultilevelROIAligner(tf_keras.layers.Layer):
  """Performs ROIAlign for the second stage processing."""

  def __init__(self, crop_size: int = 7, sample_offset: float = 0.5, **kwargs):
    """Initializes a ROI aligner.

    Args:
      crop_size: An `int` of the output size of the cropped features.
      sample_offset: A `float` in [0, 1] of the subpixel sample offset.
      **kwargs: Additional keyword arguments passed to Layer.
    """
    self._config_dict = {
        'crop_size': crop_size,
        'sample_offset': sample_offset,
    }
    super(MultilevelROIAligner, self).__init__(**kwargs)

  def call(self,
           features: Mapping[str, tf.Tensor],
           boxes: tf.Tensor,
           training: bool = None):
    """Generates ROIs.

    Args:
      features: A dictionary with key as pyramid level and value as features.
        The features are in shape of
        [batch_size, height_l, width_l, num_filters].
      boxes: A 3-D `tf.Tensor` of shape [batch_size, num_boxes, 4]. Each row
        represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
        from grid point.
      training: A `bool` of whether it is in training mode.

    Returns:
      A 5-D `tf.Tensor` representing feature crop of shape
      [batch_size, num_boxes, crop_size, crop_size, num_filters].
    """
    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features,
        boxes,
        output_size=self._config_dict['crop_size'],
        sample_offset=self._config_dict['sample_offset'])
    return roi_features

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
