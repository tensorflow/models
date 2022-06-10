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

"""Parser for video and label datasets."""

from typing import Dict, Optional, Tuple

from absl import logging
import tensorflow as tf
from official.projects.video_ssl.configs import video_ssl as exp_cfg
from official.projects.video_ssl.ops import video_ssl_preprocess_ops
from official.vision.dataloaders import video_input
from official.vision.ops import preprocess_ops_3d

IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'
Decoder = video_input.Decoder


def _process_image(image: tf.Tensor,
                   is_training: bool = True,
                   is_ssl: bool = False,
                   num_frames: int = 32,
                   stride: int = 1,
                   num_test_clips: int = 1,
                   min_resize: int = 256,
                   crop_size: int = 224,
                   num_crops: int = 1,
                   zero_centering_image: bool = False,
                   seed: Optional[int] = None) -> tf.Tensor:
  """Processes a serialized image tensor.

  Args:
    image: Input Tensor of shape [timesteps] and type tf.string of serialized
      frames.
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    is_ssl: Whether or not in self-supervised pre-training mode.
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    num_crops: Number of crops to perform on the resized frames.
    zero_centering_image: If True, frames are normalized to values in [-1, 1].
      If False, values in [0, 1].
    seed: A deterministic seed to use when sampling.

  Returns:
    Processed frames. Tensor of shape
      [num_frames * num_test_clips, crop_size, crop_size, 3].
  """
  # Validate parameters.
  if is_training and num_test_clips != 1:
    logging.warning(
        '`num_test_clips` %d is ignored since `is_training` is `True`.',
        num_test_clips)

  # Temporal sampler.
  if is_training:
    # Sampler for training.
    if is_ssl:
      # Sample two clips from linear decreasing distribution.
      image = video_ssl_preprocess_ops.sample_ssl_sequence(
          image, num_frames, True, stride)
    else:
      # Sample random clip.
      image = preprocess_ops_3d.sample_sequence(image, num_frames, True, stride)

  else:
    # Sampler for evaluation.
    if num_test_clips > 1:
      # Sample linspace clips.
      image = preprocess_ops_3d.sample_linspace_sequence(image, num_test_clips,
                                                         num_frames, stride)
    else:
      # Sample middle clip.
      image = preprocess_ops_3d.sample_sequence(image, num_frames, False,
                                                stride)

  # Decode JPEG string to tf.uint8.
  image = preprocess_ops_3d.decode_jpeg(image, 3)

  if is_training:
    # Standard image data augmentation: random resized crop and random flip.
    if is_ssl:
      image_1, image_2 = tf.split(image, num_or_size_splits=2, axis=0)
      image_1 = preprocess_ops_3d.random_crop_resize(
          image_1, crop_size, crop_size, num_frames, 3, (0.5, 2), (0.3, 1))
      image_1 = preprocess_ops_3d.random_flip_left_right(image_1, seed)
      image_2 = preprocess_ops_3d.random_crop_resize(
          image_2, crop_size, crop_size, num_frames, 3, (0.5, 2), (0.3, 1))
      image_2 = preprocess_ops_3d.random_flip_left_right(image_2, seed)

    else:
      image = preprocess_ops_3d.random_crop_resize(
          image, crop_size, crop_size, num_frames, 3, (0.5, 2), (0.3, 1))
      image = preprocess_ops_3d.random_flip_left_right(image, seed)
  else:
    # Resize images (resize happens only if necessary to save compute).
    image = preprocess_ops_3d.resize_smallest(image, min_resize)
    # Three-crop of the frames.
    image = preprocess_ops_3d.crop_image(image, crop_size, crop_size, False,
                                         num_crops)

  # Cast the frames in float32, normalizing according to zero_centering_image.
  if is_training and is_ssl:
    image_1 = preprocess_ops_3d.normalize_image(image_1, zero_centering_image)
    image_2 = preprocess_ops_3d.normalize_image(image_2, zero_centering_image)

  else:
    image = preprocess_ops_3d.normalize_image(image, zero_centering_image)

  # Self-supervised pre-training augmentations.
  if is_training and is_ssl:
    if zero_centering_image:
      image_1 = 0.5 * (image_1 + 1.0)
      image_2 = 0.5 * (image_2 + 1.0)
    # Temporally consistent color jittering.
    image_1 = video_ssl_preprocess_ops.random_color_jitter_3d(image_1)
    image_2 = video_ssl_preprocess_ops.random_color_jitter_3d(image_2)
    # Temporally consistent gaussian blurring.
    image_1 = video_ssl_preprocess_ops.random_blur(image_1, crop_size,
                                                   crop_size, 1.0)
    image_2 = video_ssl_preprocess_ops.random_blur(image_2, crop_size,
                                                   crop_size, 0.1)
    image_2 = video_ssl_preprocess_ops.random_solarization(image_2)
    image = tf.concat([image_1, image_2], axis=0)
    image = tf.clip_by_value(image, 0., 1.)
    if zero_centering_image:
      image = 2 * (image - 0.5)

  return image


def _postprocess_image(image: tf.Tensor,
                       is_training: bool = True,
                       is_ssl: bool = False,
                       num_frames: int = 32,
                       num_test_clips: int = 1,
                       num_test_crops: int = 1) -> tf.Tensor:
  """Processes a batched Tensor of frames.

  The same parameters used in process should be used here.

  Args:
    image: Input Tensor of shape [batch, timesteps, height, width, 3].
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    is_ssl: Whether or not in self-supervised pre-training mode.
    num_frames: Number of frames per subclip.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    num_test_crops: Number of test crops (1 by default). If more than 1, there
      are multiple crops for each clip at test time. If 1, there is a single
      central crop. The crops are aggreagated in the batch dimension.

  Returns:
    Processed frames. Tensor of shape
      [batch * num_test_clips * num_test_crops, num_frames, height, width, 3].
  """
  if is_ssl and is_training:
    # In this case, two clips of self-supervised pre-training are merged
    # together in batch dimenstion which will be 2 * batch.
    image = tf.concat(tf.split(image, num_or_size_splits=2, axis=1), axis=0)

  num_views = num_test_clips * num_test_crops
  if num_views > 1 and not is_training:
    # In this case, multiple views are merged together in batch dimenstion which
    # will be batch * num_views.
    image = tf.reshape(image, [-1, num_frames] + image.shape[2:].as_list())

  return image


def _process_label(label: tf.Tensor,
                   one_hot_label: bool = True,
                   num_classes: Optional[int] = None) -> tf.Tensor:
  """Processes label Tensor."""
  # Validate parameters.
  if one_hot_label and not num_classes:
    raise ValueError(
        '`num_classes` should be given when requesting one hot label.')

  # Cast to tf.int32.
  label = tf.cast(label, dtype=tf.int32)

  if one_hot_label:
    # Replace label index by one hot representation.
    label = tf.one_hot(label, num_classes)
    if len(label.shape.as_list()) > 1:
      label = tf.reduce_sum(label, axis=0)
    if num_classes == 1:
      # The trick for single label.
      label = 1 - label

  return label


class Parser(video_input.Parser):
  """Parses a video and label dataset."""

  def __init__(self,
               input_params: exp_cfg.DataConfig,
               image_key: str = IMAGE_KEY,
               label_key: str = LABEL_KEY):
    super(Parser, self).__init__(input_params, image_key, label_key)
    self._is_ssl = input_params.is_ssl

  def _parse_train_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for training."""
    # Process image and label.
    image = decoded_tensors[self._image_key]
    image = _process_image(
        image=image,
        is_training=True,
        is_ssl=self._is_ssl,
        num_frames=self._num_frames,
        stride=self._stride,
        num_test_clips=self._num_test_clips,
        min_resize=self._min_resize,
        crop_size=self._crop_size,
        zero_centering_image=self._zero_centering_image)
    image = tf.cast(image, dtype=self._dtype)
    features = {'image': image}

    label = decoded_tensors[self._label_key]
    label = _process_label(label, self._one_hot_label, self._num_classes)

    return features, label

  def _parse_eval_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for evaluation."""
    image = decoded_tensors[self._image_key]
    image = _process_image(
        image=image,
        is_training=False,
        num_frames=self._num_frames,
        stride=self._stride,
        num_test_clips=self._num_test_clips,
        min_resize=self._min_resize,
        crop_size=self._crop_size,
        num_crops=self._num_crops,
        zero_centering_image=self._zero_centering_image)
    image = tf.cast(image, dtype=self._dtype)
    features = {'image': image}

    label = decoded_tensors[self._label_key]
    label = _process_label(label, self._one_hot_label, self._num_classes)

    if self._output_audio:
      audio = decoded_tensors[self._audio_feature]
      audio = tf.cast(audio, dtype=self._dtype)
      audio = preprocess_ops_3d.sample_sequence(
          audio, 20, random=False, stride=1)
      audio = tf.ensure_shape(audio, [20, 2048])
      features['audio'] = audio

    return features, label

  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.

    Args:
      is_training: a `bool` to indicate whether it is in training mode.

    Returns:
      parse: a `callable` that takes the serialized examle and generate the
        images, labels tuple where labels is a dict of Tensors that contains
        labels.
    """
    def parse(decoded_tensors):
      """Parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      else:
        return self._parse_eval_data(decoded_tensors)

    return parse


class PostBatchProcessor(object):
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self._is_training = input_params.is_training
    self._is_ssl = input_params.is_ssl
    self._num_frames = input_params.feature_shape[0]
    self._num_test_clips = input_params.num_test_clips
    self._num_test_crops = input_params.num_test_crops

  def __call__(self, features: Dict[str, tf.Tensor],
               label: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses a single tf.Example into image and label tensors."""
    for key in ['image', 'audio']:
      if key in features:
        features[key] = _postprocess_image(
            image=features[key],
            is_training=self._is_training,
            is_ssl=self._is_ssl,
            num_frames=self._num_frames,
            num_test_clips=self._num_test_clips,
            num_test_crops=self._num_test_crops)

    return features, label
