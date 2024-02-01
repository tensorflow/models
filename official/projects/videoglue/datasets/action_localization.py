# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""The dataset factory for the video action localization."""
import functools
import os
from typing import Any, Dict, List, Mapping, Optional, Union

from dmvr import video_dataset
import tensorflow as tf, tf_keras

from official.projects.videoglue.datasets.common import utils


class ActionLocalizationBaseFactory(video_dataset.BaseVideoDatasetFactory):
  """Action Localization dataset factory."""

  _BASE_DIR = '/tmp'
  _TABLES = {
      'train': 'example.tfrecord',
      'test': 'example.tfrecord',
  }
  _SUBSETS = ('train', 'test')

  _KEYFRAME_INDEX_KEY = 'clip/key_frame/frame_index'
  _GT_PREFIX = 'clip/key_frame'
  _DETECTOR_PREFIX = 'centernet'
  _ZERO_BASED_INDEX = False  # whether the labels are 0-indexed in the table.
  # threshold to be applied for filtering detected boxes.
  _TRAIN_DETECTION_SCORE = 0.9
  _EVAL_DETECTION_SCORE = 0.8

  _NUM_CLASSES = 80

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

  def _build(self,
             is_training: bool = True,
             # Video related parameters.
             num_frames: int = 32,
             temporal_stride: int = 1,
             num_instance_per_frame: int = 5,
             # Image related parameters.
             min_resize: int = 224,
             crop_size: int = 200,
             zero_centering_image: bool = False,
             color_augmentation: bool = False,
             augmentation_type: str = 'AVA',
             augmentation_params: Optional[Mapping[str, Any]] = None,
             # Test related parameters,
             num_test_clips: int = 1,
             # Label related parameters.
             one_hot_label: bool = True,
             merge_multi_labels: bool = False,
             import_detected_bboxes: bool = False):
    """Builds the data processing graph.

    Args:
      is_training: Whether or not in training mode. If `True`, random sample,
        crop and left right flip are used.
      num_frames: Number of frames per subclip.
      temporal_stride: Temporal stride to sample frames.
      num_instance_per_frame: The max number of instances per frame to keep.
      min_resize: Frames are resized so that `min(height, width)` is
        `min_resize`.
      crop_size: Final size of the frame after cropping the resized frames. Both
        height and width are the same.
      zero_centering_image: If `True`, frames are normalized to values in
        [-1, 1]. If `False`, values in [0, 1].
      color_augmentation: Whether to apply color augmentation on video clips.
      augmentation_type: The data augmentation style applied on images.
      augmentation_params: A dictionary of params for data augmentation.
      num_test_clips: Number of test clips (1 by default). If more than 1, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
        The clips are aggreagated in the batch dimension.
      one_hot_label: Whether to return one-hot label.
      merge_multi_labels: Whether to merge multi_labels.
      import_detected_bboxes: Whether to parse and return detected boxes.
    """
    if num_test_clips != 1:
      raise ValueError('only support num_test_clips = 1 for action '
                       'localization task. ')

    # Parse keyframe index.
    self.parser_builder.parse_feature(
        feature_type=tf.io.FixedLenFeature(1, dtype=tf.int64),
        feature_name=self._KEYFRAME_INDEX_KEY,
        output_name='keyframe_index',
        is_context=True)
    # Add keyframe boxes.
    for dim in ['ymin', 'xmin', 'ymax', 'xmax']:
      utils.add_context_box_dim(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          preprocessor_builder=self.preprocessor_builder,
          input_box_dim_name=f'{self._GT_PREFIX}/bbox/{dim}',
          output_box_dim_name=f'instances_{dim}',
          num_instances_per_frame=num_instance_per_frame,
          num_frames=num_frames)   # > 1 to duplicate keyframe boxes to all.
    utils.group_instance_box_dims(
        preprocessor_builder=self.preprocessor_builder,
        output_position_name='instances_position',
        box_key_prefix='instances')
    # Add boxes scores
    utils.add_context_box_dim(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        preprocessor_builder=self.preprocessor_builder,
        input_box_dim_name=f'{self._GT_PREFIX}/bbox/score',
        output_box_dim_name='instances_score',
        num_instances_per_frame=num_instance_per_frame,
        num_frames=num_frames,   # duplicate keyframe boxes to all frames.
        default_value=1.0)
    if is_training:
      filter_instances_position_by_score_fn = functools.partial(
          utils.filter_instances_box_by_score,
          box_key_prefix='instances',
          score_threshold=self._TRAIN_DETECTION_SCORE)
      self.preprocessor_builder.add_fn(
          fn=filter_instances_position_by_score_fn,
          fn_name='filter_training_boxes')
    else:
      self.preprocessor_builder.add_fn(
          fn=lambda x: utils.infer_instances_mask_from_position(inputs=x),
          fn_name='infer_instances_mask')

    # Add detected boxes.
    if (not is_training) and import_detected_bboxes:
      for dim in ['ymin', 'xmin', 'ymax', 'xmax', 'score']:
        utils.add_instance_box_dim(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            preprocessor_builder=self.preprocessor_builder,
            input_box_dim_name='{}/bbox/{}'.format(self._DETECTOR_PREFIX, dim),
            output_box_dim_name='detected_instances_{}'.format(dim),
            sample_around_keyframe=True,
            sample_random=False,
            num_instances_per_frame=num_instance_per_frame,
            num_frames=num_frames,
            temporal_stride=temporal_stride,
            sync_random_state=True)
      utils.group_instance_box_dims(
          preprocessor_builder=self.preprocessor_builder,
          box_key_prefix='detected_instances',
          output_position_name='detected_instances_position')
      filter_instances_position_by_score_fn = functools.partial(
          utils.filter_instances_box_by_score,
          box_key_prefix='detected_instances',
          score_threshold=self._EVAL_DETECTION_SCORE)
      self.preprocessor_builder.add_fn(
          fn=filter_instances_position_by_score_fn,
          fn_name='filter_detected_boxes')

    # Add images.
    utils.add_image(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        postprocessor_builder=self.postprocessor_builder,
        sample_around_keyframe=True,
        is_training=is_training,
        num_frames=num_frames,
        temporal_stride=temporal_stride,
        num_test_clips=num_test_clips,
        crop_size=crop_size,
        min_resize=min_resize,
        multi_crop=False,
        zero_centering_image=zero_centering_image,
        augmentation_type=augmentation_type,
        augmentation_params=augmentation_params,
        sync_random_state=True)

    # Adapt boxes to the image augmentations.
    utils.adjust_positions(
        preprocessor_builder=self.preprocessor_builder,
        input_tensor_name='instances_position',
        output_tensor_name='instances_position')
    if import_detected_bboxes:
      utils.adjust_positions(
          preprocessor_builder=self.preprocessor_builder,
          input_tensor_name='detected_instances_position',
          output_tensor_name='detected_instances_position')

    utils.add_context_label(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        preprocessor_builder=self.preprocessor_builder,
        input_label_index_feature_name=f'{self._GT_PREFIX}/bbox/label/index',
        input_label_name_feature_name=f'{self._GT_PREFIX}/bbox/label/string',
        num_instances_per_frame=num_instance_per_frame,
        num_frames=num_frames,
        zero_based_index=self._ZERO_BASED_INDEX,
        # merge_multi_labels fn expects label in 0-index id.
        one_hot_label=False if merge_multi_labels else one_hot_label,
        num_classes=self._NUM_CLASSES,
        add_label_name=False)
    if merge_multi_labels:
      self.preprocessor_builder.add_fn(
          fn=functools.partial(
              utils.merge_multi_labels,
              num_classes=self._NUM_CLASSES),
          fn_name='merge_multi_labels')

    if is_training and color_augmentation:
      utils.apply_default_color_augmentations(
          preprocessor_builder=self.preprocessor_builder,
          zero_centering_image=zero_centering_image)

    self.postprocessor_builder.add_fn(
        fn=utils.update_valid_instances_mask,
        fn_name='update_valid_instances_mask')

    select_keyframe_instances_fn = functools.partial(
        self._select_keyframe_instances,
        keyframe_index=(num_frames // 2),
        import_detected_bboxes=import_detected_bboxes)
    self.postprocessor_builder.add_fn(
        fn=select_keyframe_instances_fn,
        fn_name='slice_keyframe_instances')

  def _select_keyframe_instances(
      self,
      inputs: Dict[str, tf.Tensor],
      keyframe_index: int,
      import_detected_bboxes: bool) -> Dict[str, tf.Tensor]:
    """Slices only instance-related inputs on keyframes.

    Args:
      inputs: The inputs dictionary containing instance related tensors.
        Tensors' rank should be >= 3, with the order of
        [batch, time, instances, ...].
      keyframe_index: The integar local index for the keyframe. Typically this
        is the middle frame in the 5D tensor.
      import_detected_bboxes: Whether the pipeline has imported the detected
        instances boxes.

    Returns:
      The returned dictionary containing only keyframe instances inputs.
    """
    instances_name_list = [
        'label', 'instances_position', 'instances_score', 'instances_mask',
        'nonmerge_label', 'nonmerge_instances_position'
    ]
    if import_detected_bboxes:
      instances_name_list += [
          'detected_instances_position', 'detected_instances_score',
          'detected_instances_mask'
      ]
    for name in instances_name_list:
      tensor = inputs[name][:, keyframe_index, ...]
      inputs[name] = tensor
    return inputs

  def tables(self) -> Mapping[str, Union[str, List[str]]]:
    """Returns a dictionary from table name to relative path."""
    return self._TABLES


class AVAKineticsFactory(ActionLocalizationBaseFactory):
  """AVA-Kinetics data reader."""

  _BASE_DIR = '/abc'
  _TABLES = {
      'train': 'tfse_avakinetics-train.tfrecord@1000',
      'test': 'tfse_avakinetics-val.tfrecord@1000',
  }
  _SPLITS = ('train', 'test')

  # In AVA-K we use centernet detector. Choose lower score threshold for eval.
  _TRAIN_DETECTION_SCORE = 0.9
  _EVAL_DETECTION_SCORE = 0.2


class AVAFactory(ActionLocalizationBaseFactory):
  """AVA v2.2 data reader."""

  _BASE_DIR = '/abc'
  _TABLES = {
      'train': 'tfse_ava_v2.2_train.tfrecord@1000',
      'test': 'tfse_ava_v2.2_val.tfrecord@1000',
  }
  _SPLITS = ('train', 'test')

  _KEYFRAME_INDEX_KEY = 'key_frame/frame_index'
  _DETECTOR_PREFIX = 'detectron_frcnn_person/region'
  _ZERO_BASED_INDEX = True
