# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Parser for video and label datasets."""

from typing import Dict, Optional, Tuple

from absl import logging
import tensorflow as tf

from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops_3d

IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'


def _process_image(image: tf.Tensor,
                   is_training: bool = True,
                   num_frames: int = 32,
                   stride: int = 1,
                   num_test_clips: int = 1,
                   min_resize: int = 224,
                   crop_size: int = 200,
                   zero_centering_image: bool = False,
                   seed: Optional[int] = None) -> tf.Tensor:
  """Processes a serialized image tensor.

  Args:
    image: Input Tensor of shape [timesteps] and type tf.string of serialized
      frames.
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
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
    # Sample random clip.
    image = preprocess_ops_3d.sample_sequence(image, num_frames, True, stride,
                                              seed)
  elif num_test_clips > 1:
    # Sample linspace clips.
    image = preprocess_ops_3d.sample_linspace_sequence(image, num_test_clips,
                                                       num_frames, stride)
  else:
    # Sample middle clip.
    image = preprocess_ops_3d.sample_sequence(image, num_frames, False, stride)

  # Decode JPEG string to tf.uint8.
  image = preprocess_ops_3d.decode_jpeg(image, 3)

  # Resize images (resize happens only if necessary to save compute).
  image = preprocess_ops_3d.resize_smallest(image, min_resize)

  if is_training:
    # Standard image data augmentation: random crop and random flip.
    image = preprocess_ops_3d.crop_image(image, crop_size, crop_size, True,
                                         seed)
    image = preprocess_ops_3d.random_flip_left_right(image, seed)
  else:
    # Central crop of the frames.
    image = preprocess_ops_3d.crop_image(image, crop_size, crop_size, False)

  # Cast the frames in float32, normalizing according to zero_centering_image.
  return preprocess_ops_3d.normalize_image(image, zero_centering_image)


def _postprocess_image(image: tf.Tensor,
                       is_training: bool = True,
                       num_frames: int = 32,
                       num_test_clips: int = 1) -> tf.Tensor:
  """Processes a batched Tensor of frames.

  The same parameters used in process should be used here.

  Args:
    image: Input Tensor of shape [batch, timesteps, height, width, 3].
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.

  Returns:
    Processed frames. Tensor of shape
      [batch * num_test_clips, num_frames, height, width, 3].
  """
  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimenstion which
    # will be B * num_test_clips.
    image = tf.reshape(
        image, (-1, num_frames, image.shape[2], image.shape[3], image.shape[4]))

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

  return label


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self, image_key: str = IMAGE_KEY, label_key: str = LABEL_KEY):
    self._image_key = IMAGE_KEY
    self._label_key = LABEL_KEY
    self._context_description = {
        # One integer stored in context.
        self._label_key: tf.io.FixedLenFeature((), tf.int64),
    }
    self._sequence_description = {
        # Each image is a string encoding JPEG.
        self._image_key: tf.io.FixedLenSequenceFeature((), tf.string),
    }

  def decode(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    context, sequences = tf.io.parse_single_sequence_example(
        serialized_example, self._context_description,
        self._sequence_description)
    return {
        self._image_key: sequences[self._image_key],
        self._label_key: context[self._label_key]
    }


class Parser(parser.Parser):
  """Parses a video and label dataset."""

  def __init__(self,
               input_params: exp_cfg.DataConfig,
               image_key: str = IMAGE_KEY,
               label_key: str = LABEL_KEY):
    self._num_frames = input_params.feature_shape[0]
    self._stride = input_params.temporal_stride
    self._num_test_clips = input_params.num_test_clips
    self._min_resize = input_params.min_image_size
    self._crop_size = input_params.feature_shape[1]
    self._one_hot_label = input_params.one_hot
    self._num_classes = input_params.num_classes
    self._image_key = image_key
    self._label_key = label_key

  def _parse_train_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for training."""
    # Process image and label.
    image = decoded_tensors[self._image_key]
    label = decoded_tensors[self._label_key]
    image = _process_image(
        image=image,
        is_training=True,
        num_frames=self._num_frames,
        stride=self._stride,
        num_test_clips=self._num_test_clips,
        min_resize=self._min_resize,
        crop_size=self._crop_size)
    label = _process_label(label, self._one_hot_label, self._num_classes)

    return {'image': image}, label

  def _parse_eval_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for evaluation."""
    image = decoded_tensors[self._image_key]
    label = decoded_tensors[self._label_key]
    image = _process_image(
        image=image,
        is_training=False,
        num_frames=self._num_frames,
        stride=self._stride,
        num_test_clips=self._num_test_clips,
        min_resize=self._min_resize,
        crop_size=self._crop_size)
    label = _process_label(label, self._one_hot_label, self._num_classes)

    return {'image': image}, label


class PostBatchProcessor(object):
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self._is_training = input_params.is_training

    self._num_frames = input_params.feature_shape[0]
    self._num_test_clips = input_params.num_test_clips

  def __call__(
      self,
      image: Dict[str, tf.Tensor],
      label: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses a single tf.Example into image and label tensors."""
    image = image['image']
    image = _postprocess_image(
        image=image,
        is_training=self._is_training,
        num_frames=self._num_frames,
        num_test_clips=self._num_test_clips)

    return {'image': image}, label
