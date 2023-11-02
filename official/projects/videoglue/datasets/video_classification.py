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

"""Thd video classification dataset generator."""
import os
from typing import Any, Mapping, Optional

from dmvr import video_dataset
import tensorflow as tf, tf_keras

from official.projects.videoglue.datasets.common import utils
from official.vision.ops import augment


class VideoClassificationBaseFactory(video_dataset.BaseVideoDatasetFactory):
  """VideoClassification dataset factory."""

  _BASE_DIR = '/tmp'

  _TABLES = {
      'train': 'sample.tfrecord',
      'test': 'sample.tfrecord',
  }

  _SUBSETS = ('train', 'test')

  _NUM_CLASSES = 400
  _INPUT_LABEL_INDEX_KEY = 'clip/label/index'
  _OUTPUT_LABEL_INDEX_KEY = 'label'

  def __init__(
      self,
      subset: str = 'train'):
    """Initializes the factory."""

    if subset not in self._SUBSETS:
      raise ValueError(f'Invalid subset "{subset}".'
                       f' The available subsets are: {self._SUBSETS}')

    table_name = os.path.join(self._BASE_DIR, self._TABLES[subset])
    shards = utils.get_shards(table_name)

    super().__init__(shards)

  def _build(
      self,
      is_training: bool = True,
      # Video related parameters.
      num_frames: int = 32,
      temporal_stride: int = 1,
      sample_from_segments: bool = False,
      # Image related parameters.
      min_resize: int = 256,
      crop_size: int = 224,
      zero_centering_image: bool = False,
      random_flip_image: bool = True,
      augmentation_type: str = 'VGG',
      augmentation_params: Optional[Mapping[str, Any]] = None,
      randaug_params: Optional[Mapping[str, Any]] = None,
      autoaug_params: Optional[Mapping[str, Any]] = None,
      mixup_cutmix_params: Optional[Mapping[str, Any]] = None,
      # Test related parameters,
      num_test_clips: int = 1,
      multi_crop: bool = False,
      # Label related parameters.
      one_hot_label: bool = True,
      get_label_str: bool = False):
    """Default builder for this dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      temporal_stride: temporal stride to sample frames.
      sample_from_segments: Whether to sample frames from segments of a video.
        If True, the temporal_stride and num_test_clips will be ignored.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      zero_centering_image: whether to have images between [-1, 1] or [0, 1].
      random_flip_image: If True, frames are randomly horizontal flipped during
        the training.
      augmentation_type: The data augmentation style applied on images.
      augmentation_params: A dictionary of params for data augmentation.
      randaug_params: A dictionary of params for RandAug policy.
      autoaug_params:  A dictionary of params for AutoAug policy.
      mixup_cutmix_params: A dictionary of params for Mixup and Cutmix data
        augmentation policy.
      num_test_clips: number of test clip (1 by default). If more than one, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
      multi_crop: if True, 3 crops will be sampled from one video clip.
      one_hot_label: whether or not to return one hot version of labels.
      get_label_str: whether or not to return label as text.
    """
    utils.add_image(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        postprocessor_builder=self.postprocessor_builder,
        input_feature_name='image/encoded',
        is_training=is_training,
        num_frames=num_frames,
        temporal_stride=temporal_stride,
        sample_from_segments=sample_from_segments,
        num_test_clips=num_test_clips,
        multi_crop=multi_crop,
        min_resize=min_resize,
        crop_size=crop_size,
        zero_centering_image=zero_centering_image,
        random_flip_image=random_flip_image,
        augmentation_type=augmentation_type,
        augmentation_params=augmentation_params,
        randaug_params=randaug_params,
        autoaug_params=autoaug_params,
        sync_random_state=is_training)

    utils.add_label(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        one_hot_label=one_hot_label,
        input_label_index_feature_name=self._INPUT_LABEL_INDEX_KEY,
        output_label_index_feature_name=self._OUTPUT_LABEL_INDEX_KEY,
        num_classes=self._NUM_CLASSES,
        add_label_name=get_label_str)

    if is_training and mixup_cutmix_params is not None:
      def mixup_and_cutmix_fn(inputs):
        image = inputs['image']
        label = inputs['label']
        if one_hot_label:
          label = tf.math.argmax(label, axis=-1)
        augmenter = augment.MixupAndCutmix(
            mixup_alpha=mixup_cutmix_params['mixup_alpha'],
            cutmix_alpha=mixup_cutmix_params['cutmix_alpha'],
            prob=mixup_cutmix_params['prob'],
            label_smoothing=mixup_cutmix_params['label_smoothing'],
            num_classes=self._NUM_CLASSES)
        image, label = augmenter(image, label)
        inputs['image'] = image
        inputs['label'] = label
        return inputs

      self.postprocessor_builder.add_fn(
          mixup_and_cutmix_fn, fn_name='mixup_and_cutmix')

  def tables(self):
    return self._TABLES


class MomentsInTimeFactory(VideoClassificationBaseFactory):
  """Moments-in-time dataset."""

  _BASE_DIR = '/tmp'

  _TABLES = {
      'train': 'tfse_moments_in_time-train.tfrecord@1024',
      'test': 'tfse_moments_in_time-validation.tfrecord@1024',
  }
  _NUM_CLASSES = 339


class Sthv2Factory(VideoClassificationBaseFactory):
  """Sth-sth v2 dataset."""

  _BASE_DIR = '/tmp'

  _TABLES = {
      'train': 'tfse_something-v2-train.tfrecord@128',
      'test': 'tfse_something-v2-validation.tfrecord@128',
  }

  _NUM_CLASSES = 174


class Diving48Factory(VideoClassificationBaseFactory):
  """Diving48 dataset."""

  _BASE_DIR = '/tmp'

  _TABLES = {
      'train': 'tfse_diving48-train.tfrecord@200',
      'test': 'tfse_diving48-test.tfrecord@200',
  }

  _NUM_CLASSES = 48


class Kinetics400Factory(VideoClassificationBaseFactory):
  """Kinetics400 dataset."""

  _BASE_DIR = '/tmp'

  _TABLES = {
      'train': 'tfse_kinetics400-train.tfrecord@496',
      'test': 'tfse_kinetics400-val.tfrecord@141',
  }

  _NUM_CLASSES = 400
