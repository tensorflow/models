# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Parser for video and label datasets."""

from typing import Dict, Optional, Tuple, Union

from absl import logging
import tensorflow as tf

from official.vision.beta.configs import video_classification as exp_cfg
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
from official.vision.beta.ops import preprocess_ops_3d

IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'


def process_image(image: tf.Tensor,
                  is_training: bool = True,
                  num_frames: int = 32,
                  stride: int = 1,
                  random_stride_range: int = 0,
                  num_test_clips: int = 1,
                  min_resize: int = 256,
                  crop_size: int = 224,
                  num_crops: int = 1,
                  zero_centering_image: bool = False,
                  min_aspect_ratio: float = 0.5,
                  max_aspect_ratio: float = 2,
                  min_area_ratio: float = 0.49,
                  max_area_ratio: float = 1.0,
                  seed: Optional[int] = None) -> tf.Tensor:
  """Processes a serialized image tensor.

  Args:
    image: Input Tensor of shape [timesteps] and type tf.string of serialized
      frames.
    is_training: Whether or not in training mode. If True, random sample, crop
      and left right flip is used.
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    random_stride_range: An int indicating the min and max bounds to uniformly
      sample different strides from the video. E.g., a value of 1 with stride=2
      will uniformly sample a stride in {1, 2, 3} for each video in a batch.
      Only used enabled training for the purposes of frame-rate augmentation.
      Defaults to 0, which disables random sampling.
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
    min_aspect_ratio: The minimum aspect range for cropping.
    max_aspect_ratio: The maximum aspect range for cropping.
    min_area_ratio: The minimum area range for cropping.
    max_area_ratio: The maximum area range for cropping.
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

  if random_stride_range < 0:
    raise ValueError('Random stride range should be >= 0, got {}'.format(
        random_stride_range))

  # Temporal sampler.
  if is_training:
    if random_stride_range > 0:
      # Uniformly sample different frame-rates
      stride = tf.random.uniform(
          [],
          tf.maximum(stride - random_stride_range, 1),
          stride + random_stride_range,
          dtype=tf.int32)

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

  if is_training:
    # Standard image data augmentation: random resized crop and random flip.
    image = preprocess_ops_3d.random_crop_resize(
        image, crop_size, crop_size, num_frames, 3,
        (min_aspect_ratio, max_aspect_ratio),
        (min_area_ratio, max_area_ratio))
    image = preprocess_ops_3d.random_flip_left_right(image, seed)
  else:
    # Resize images (resize happens only if necessary to save compute).
    image = preprocess_ops_3d.resize_smallest(image, min_resize)
    # Crop of the frames.
    image = preprocess_ops_3d.crop_image(image, crop_size, crop_size, False,
                                         num_crops)

  # Cast the frames in float32, normalizing according to zero_centering_image.
  return preprocess_ops_3d.normalize_image(image, zero_centering_image)


def postprocess_image(image: tf.Tensor,
                      is_training: bool = True,
                      num_frames: int = 32,
                      num_test_clips: int = 1,
                      num_test_crops: int = 1) -> tf.Tensor:
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
    num_test_crops: Number of test crops (1 by default). If more than 1, there
      are multiple crops for each clip at test time. If 1, there is a single
      central crop. The crops are aggreagated in the batch dimension.

  Returns:
    Processed frames. Tensor of shape
      [batch * num_test_clips * num_test_crops, num_frames, height, width, 3].
  """
  num_views = num_test_clips * num_test_crops
  if num_views > 1 and not is_training:
    # In this case, multiple views are merged together in batch dimenstion which
    # will be batch * num_views.
    image = tf.reshape(image, [-1, num_frames] + image.shape[2:].as_list())

  return image


def process_label(label: tf.Tensor,
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


class Decoder(decoder.Decoder):
  """A tf.Example decoder for classification task."""

  def __init__(self, image_key: str = IMAGE_KEY, label_key: str = LABEL_KEY):
    self._context_description = {
        # One integer stored in context.
        label_key: tf.io.VarLenFeature(tf.int64),
    }
    self._sequence_description = {
        # Each image is a string encoding JPEG.
        image_key: tf.io.FixedLenSequenceFeature((), tf.string),
    }

  def add_feature(self, feature_name: str,
                  feature_type: Union[tf.io.VarLenFeature,
                                      tf.io.FixedLenFeature,
                                      tf.io.FixedLenSequenceFeature]):
    self._sequence_description[feature_name] = feature_type

  def add_context(self, feature_name: str,
                  feature_type: Union[tf.io.VarLenFeature,
                                      tf.io.FixedLenFeature,
                                      tf.io.FixedLenSequenceFeature]):
    self._context_description[feature_name] = feature_type

  def decode(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    result = {}
    context, sequences = tf.io.parse_single_sequence_example(
        serialized_example, self._context_description,
        self._sequence_description)
    result.update(context)
    result.update(sequences)
    for key, value in result.items():
      if isinstance(value, tf.SparseTensor):
        result[key] = tf.sparse.to_dense(value)
    return result


class Parser(parser.Parser):
  """Parses a video and label dataset."""

  def __init__(self,
               input_params: exp_cfg.DataConfig,
               image_key: str = IMAGE_KEY,
               label_key: str = LABEL_KEY):
    self._num_frames = input_params.feature_shape[0]
    self._stride = input_params.temporal_stride
    self._random_stride_range = input_params.random_stride_range
    self._num_test_clips = input_params.num_test_clips
    self._min_resize = input_params.min_image_size
    self._crop_size = input_params.feature_shape[1]
    self._num_crops = input_params.num_test_crops
    self._one_hot_label = input_params.one_hot
    self._num_classes = input_params.num_classes
    self._image_key = image_key
    self._label_key = label_key
    self._dtype = tf.dtypes.as_dtype(input_params.dtype)
    self._output_audio = input_params.output_audio
    self._min_aspect_ratio = input_params.aug_min_aspect_ratio
    self._max_aspect_ratio = input_params.aug_max_aspect_ratio
    self._min_area_ratio = input_params.aug_min_area_ratio
    self._max_area_ratio = input_params.aug_max_area_ratio
    if self._output_audio:
      self._audio_feature = input_params.audio_feature
      self._audio_shape = input_params.audio_feature_shape

  def _parse_train_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for training."""
    # Process image and label.
    image = decoded_tensors[self._image_key]
    image = process_image(
        image=image,
        is_training=True,
        num_frames=self._num_frames,
        stride=self._stride,
        random_stride_range=self._random_stride_range,
        num_test_clips=self._num_test_clips,
        min_resize=self._min_resize,
        crop_size=self._crop_size,
        min_aspect_ratio=self._min_aspect_ratio,
        max_aspect_ratio=self._max_aspect_ratio,
        min_area_ratio=self._min_area_ratio,
        max_area_ratio=self._max_area_ratio)
    image = tf.cast(image, dtype=self._dtype)
    features = {'image': image}

    label = decoded_tensors[self._label_key]
    label = process_label(label, self._one_hot_label, self._num_classes)

    if self._output_audio:
      audio = decoded_tensors[self._audio_feature]
      audio = tf.cast(audio, dtype=self._dtype)
      # TODO(yeqing): synchronize audio/video sampling. Especially randomness.
      audio = preprocess_ops_3d.sample_sequence(
          audio, self._audio_shape[0], random=False, stride=1)
      audio = tf.ensure_shape(audio, self._audio_shape)
      features['audio'] = audio

    return features, label

  def _parse_eval_data(
      self, decoded_tensors: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses data for evaluation."""
    image = decoded_tensors[self._image_key]
    image = process_image(
        image=image,
        is_training=False,
        num_frames=self._num_frames,
        stride=self._stride,
        num_test_clips=self._num_test_clips,
        min_resize=self._min_resize,
        crop_size=self._crop_size,
        num_crops=self._num_crops)
    image = tf.cast(image, dtype=self._dtype)
    features = {'image': image}

    label = decoded_tensors[self._label_key]
    label = process_label(label, self._one_hot_label, self._num_classes)

    if self._output_audio:
      audio = decoded_tensors[self._audio_feature]
      audio = tf.cast(audio, dtype=self._dtype)
      audio = preprocess_ops_3d.sample_sequence(
          audio, 20, random=False, stride=1)
      audio = tf.ensure_shape(audio, self._audio_shape)
      features['audio'] = audio

    return features, label


class PostBatchProcessor(object):
  """Processes a video and label dataset which is batched."""

  def __init__(self, input_params: exp_cfg.DataConfig):
    self._is_training = input_params.is_training

    self._num_frames = input_params.feature_shape[0]
    self._num_test_clips = input_params.num_test_clips
    self._num_test_crops = input_params.num_test_crops

  def __call__(self, features: Dict[str, tf.Tensor],
               label: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Parses a single tf.Example into image and label tensors."""
    for key in ['image']:
      if key in features:
        features[key] = postprocess_image(
            image=features[key],
            is_training=self._is_training,
            num_frames=self._num_frames,
            num_test_clips=self._num_test_clips,
            num_test_crops=self._num_test_crops)

    return features, label
