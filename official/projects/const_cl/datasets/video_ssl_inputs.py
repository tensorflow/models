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

"""Video SSL datasets."""

from typing import Dict, Tuple, Optional
import tensorflow as tf, tf_keras

from official.projects.video_ssl.dataloaders import video_ssl_input


IMAGE_KEY = video_ssl_input.IMAGE_KEY
LABEL_KEY = video_ssl_input.LABEL_KEY
Decoder = video_ssl_input.Decoder


class Parser(video_ssl_input.Parser):
  """Parses a video dataset for SSL."""

  def __init__(self,
               input_params,
               image_key: str = IMAGE_KEY,
               label_key: str = LABEL_KEY):
    super().__init__(
        input_params=input_params,
        image_key=image_key,
        label_key=label_key)

    self._num_instances = input_params.num_instances
    self._num_frames = input_params.feature_shape[0]

  def _generate_random_positions(
      self, seed: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generates random instance positions in videos."""

    num_frames = self._num_frames * 2 if self._is_ssl else self._num_frames
    shape = [num_frames, self._num_instances, 4]
    xmin = tf.random.uniform(shape=shape[:-1],
                             minval=0.0,
                             maxval=1.0,
                             dtype=tf.float32,
                             seed=seed)
    ymin = tf.random.uniform(shape=shape[:-1],
                             minval=0.0,
                             maxval=1.0,
                             dtype=tf.float32,
                             seed=seed)
    xdelta = tf.random.uniform(shape=shape[:-1],
                               minval=0.1,
                               maxval=0.5,
                               dtype=tf.float32,
                               seed=seed)
    aspect_ratio = tf.random.uniform(shape=shape[:-1],
                                     minval=0.5,
                                     maxval=2.0,
                                     dtype=tf.float32,
                                     seed=seed)
    ydelta = xdelta * aspect_ratio
    xmax = tf.math.minimum(xmin + xdelta, 1.0 - 1e-3)
    ymax = tf.math.minimum(ymin + ydelta, 1.0 - 1e-3)
    random_positions = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    random_positions = tf.cast(random_positions, dtype=self._dtype)
    instances_mask = tf.ones(shape[:-1], dtype=tf.bool)
    return random_positions, instances_mask

  def _parse_train_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for training."""
    features, label = super()._parse_train_data(decoded_tensors=decoded_tensors)
    instances_position, instances_mask = self._generate_random_positions(
        seed=1234)
    features.update({
        'instances_position': instances_position,
        'instances_mask': instances_mask,
    })
    return features, label


class PostBatchProcessor(video_ssl_input.PostBatchProcessor):
  """Processes a video and label dataset which is batched."""

  def __call__(self,
               features: Dict[str, tf.Tensor],
               label: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Postprocesses features and label tensors."""
    features, label = super().__call__(features, label)

    for key in ['instances_position', 'instances_mask']:
      if key in features and self._is_ssl and self._is_training:
        features[key] = tf.concat(
            tf.split(features[key], num_or_size_splits=2, axis=1), axis=0)

    return features, label
