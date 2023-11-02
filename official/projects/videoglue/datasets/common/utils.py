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

"""Utils for processing data."""
import functools
from typing import Any, Mapping, Optional, MutableMapping, Dict

from dmvr import builders
from dmvr import modalities
import tensorflow as tf, tf_keras

from official.projects.videoglue.datasets.common import processors
from official.vision.ops import augment
from official.vision.ops import box_ops

add_label = modalities.add_label


def pad_or_clip_nd(tensor: tf.Tensor, output_shape: list[int]):
  """Pads or clips given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


def merge_boxes_with_multiple_labels(boxes: tf.Tensor,
                                     classes: tf.Tensor,
                                     confidences: tf.Tensor,
                                     num_classes: int,
                                     quantization_bins: int = 10000):
  """Merges boxes with same coordinates and returns K-hot encoded classes.

  Args:
    boxes: A tf.float32 tensor with shape [N, 4] holding N boxes. Only
      normalized coordinates are allowed.
    classes: A tf.int32 tensor with shape [N] holding class indices.
      The class index starts at 0.
    confidences: A tf.float32 tensor with shape [N] holding class confidences.
    num_classes: total number of classes to use for K-hot encoding.
    quantization_bins: the number of bins used to quantize the box coordinate.

  Returns:
    merged_boxes: A tf.float32 tensor with shape [N', 4] holding boxes,
      where N' <= N.
    class_encodings: A tf.int32 tensor with shape [N', num_classes] holding
      K-hot encodings for the merged boxes.
    confidence_encodings: A tf.float32 tensor with shape [N', num_classes]
      holding encodings of confidences for the merged boxes.
    merged_box_indices: A tf.int32 tensor with shape [N'] holding original
      indices of the boxes.
  """

  def _assert_shape_equal_along_first_dimension(shape_a, shape_b):
    """Asserts that shape_a and shape_b are the same along the 0th-dimension.

    If the shapes are static, raises a ValueError when the shapes
    mismatch.

    If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
    mismatch.

    Args:
      shape_a: a list containing shape of the first tensor.
      shape_b: a list containing shape of the second tensor.

    Returns:
      Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
      when the shapes are dynamic.

    Raises:
      ValueError: When shapes are both static and unequal.
    """
    if isinstance(shape_a[0], int) and isinstance(shape_b[0], int):
      if shape_a[0] != shape_b[0]:
        raise ValueError('Unequal first dimension {}, {}'.format(
            shape_a[0], shape_b[0]))
      else: return tf.no_op()
    else:
      return tf.assert_equal(shape_a[0], shape_b[0])

  def _assert_box_normalized(boxes, maximum_normalized_coordinate=1.1):
    """Asserts the input box tensor is normalized.

    Args:
      boxes: a tensor of shape [N, 4] where N is the number of boxes.
      maximum_normalized_coordinate: Maximum coordinate value to be considered
        as normalized, default to 1.1.

    Returns:
      a tf.Assert op which fails when the input box tensor is not normalized.

    Raises:
      ValueError: When the input box tensor is not normalized.
    """
    box_minimum = tf.reduce_min(boxes)
    box_maximum = tf.reduce_max(boxes)
    return tf.Assert(
        tf.logical_and(
            tf.less_equal(box_maximum, maximum_normalized_coordinate),
            tf.greater_equal(box_minimum, 0)),
        [boxes])

  boxes_shape = tf.shape(boxes)
  classes_shape = tf.shape(classes)
  confidences_shape = tf.shape(confidences)
  box_class_shape_assert = _assert_shape_equal_along_first_dimension(
      boxes_shape, classes_shape)
  box_confidence_shape_assert = _assert_shape_equal_along_first_dimension(
      boxes_shape, confidences_shape)
  box_dimension_assert = tf.assert_equal(boxes_shape[1], 4)
  box_normalized_assert = _assert_box_normalized(boxes)

  with tf.control_dependencies(
      [box_class_shape_assert, box_confidence_shape_assert,
       box_dimension_assert, box_normalized_assert]):
    quantized_boxes = tf.cast(boxes * (quantization_bins - 1), tf.int64)
    ymin, xmin, ymax, xmax = tf.unstack(quantized_boxes, axis=1)
    hashcodes = (
        ymin +
        xmin * quantization_bins +
        ymax * quantization_bins * quantization_bins +
        xmax * quantization_bins * quantization_bins * quantization_bins)
    unique_hashcodes, unique_indices = tf.unique(hashcodes)
    num_boxes = tf.shape(boxes)[0]
    num_unique_boxes = tf.shape(unique_hashcodes)[0]
    merged_box_indices = tf.math.unsorted_segment_min(
        tf.range(num_boxes), unique_indices, num_unique_boxes)
    merged_boxes = tf.gather(boxes, merged_box_indices)
    unique_indices = tf.cast(unique_indices, tf.int64)
    classes = tf.cast(classes, tf.int64)

    def map_box_encodings(i):
      """Produces box K-hot and score encodings for each class index."""
      box_mask = tf.equal(
          unique_indices, i * tf.ones(num_boxes, dtype=tf.int64))
      box_mask = tf.reshape(box_mask, [-1])
      box_indices = tf.boolean_mask(classes, box_mask)
      box_confidences = tf.boolean_mask(confidences, box_mask)
      box_class_encodings = tf.compat.v1.sparse_to_dense(
          box_indices, [num_classes], tf.constant(1, dtype=tf.int64),
          validate_indices=False)
      box_confidence_encodings = tf.compat.v1.sparse_to_dense(
          box_indices, [num_classes], box_confidences, validate_indices=False)
      return box_class_encodings, box_confidence_encodings

    # Important to avoid int32 here since there is no GPU kernel for int32.
    # int64 and float32 are fine.
    class_encodings, confidence_encodings = tf.map_fn(
        map_box_encodings,
        tf.range(tf.cast(num_unique_boxes, tf.int64)),
        back_prop=False,
        dtype=(tf.int64, tf.float32))

    merged_boxes = tf.reshape(merged_boxes, [-1, 4])
    class_encodings = tf.cast(class_encodings, dtype=tf.int32)
    class_encodings = tf.reshape(class_encodings, [-1, num_classes])
    confidence_encodings = tf.reshape(confidence_encodings, [-1, num_classes])
    merged_box_indices = tf.reshape(merged_box_indices, [-1])
    return (merged_boxes, class_encodings, confidence_encodings,
            merged_box_indices)


def add_image(parser_builder: builders.BaseParserBuilder,
              sampler_builder: builders.SamplerBuilder,
              decoder_builder: builders.DecoderBuilder,
              preprocessor_builder: builders.PreprocessorBuilder,
              postprocessor_builder: builders.PostprocessorBuilder,
              input_feature_name: str = 'image/encoded',
              output_feature_name: str = builders.IMAGE_FEATURE_NAME,
              is_training: bool = True,
              sample_around_keyframe: bool = False,
              sample_from_segments: bool = False,
              # Video related parameters.
              num_frames: int = 32,
              temporal_stride: int = 1,
              num_test_clips: int = 1,
              crop_size: int = 200,
              min_resize: int = 224,
              multi_crop: bool = False,
              zero_centering_image: bool = False,
              random_flip_image: bool = True,
              augmentation_type: str = 'Inception',
              augmentation_params: Optional[Mapping[str, Any]] = None,
              randaug_params: Optional[Mapping[str, Any]] = None,
              autoaug_params: Optional[Mapping[str, Any]] = None,
              sync_random_state: bool = True,
              seed: Optional[int] = None):
  """Adds functions to process image feature to builders.

  Args:
    parser_builder: An instance of a builders.BaseParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    decoder_builder: An instance of a builders.DecoderBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    postprocessor_builder: An instance of a builders.PostprocessorBuilder.
    input_feature_name: Name of the feature in the input SequenceExample.
      Exposing this as an argument allows using this function for different
      image features.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      image features.
    is_training: Whether or not perform random operations. If True, random
      sample, crop and left right flip is used.
    sample_around_keyframe: Whether to sample clip around the keyframe. If True,
      the random temporal sampling will be overridden and disabled.
    sample_from_segments: Whether to sample frames from segments of a video. If
      True, the temporal_stride will be ignored.
    num_frames: Number of frames per subclip.
    temporal_stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    min_resize: The minimal length resize before cropping.
    multi_crop: Whether to perform 3-view crop or not. This is only enabled
      in evaluation mode. If is_training=True, this is ignored.
    zero_centering_image: If True, frames are normalized to values in [-1, 1].
      If False, values in [0, 1].
    random_flip_image: If True, frames are randomly horizontal flipped during
      the training.
    augmentation_type: The style of Crop+Resize procedure. Support options:
      ['Inception', 'VGG'].
    augmentation_params: A dictionary contains image augmentation parameters
      associated with the augmentation style.
    randaug_params: A dictionary of params for RandAug policy.
    autoaug_params:  A dictionary of params for AutoAug policy.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      True will use the same outcome in random operations such as sampling and
      cropping.
    seed: the random seed.
  """
  # Validate parameters.
  if sync_random_state and multi_crop:
    raise ValueError('multi_crop is not supported with sync random states.')

  if augmentation_type.lower() == 'ava' and multi_crop:
    raise ValueError('multi_crop should not be combined with ava augmentation.')

  if sample_from_segments and sample_around_keyframe:
    raise ValueError('sample_from_segments and sample_around_keyframes cannot '
                     'be True at the same time.')

  if sample_from_segments and num_test_clips > 1:
    raise ValueError(
        'sample_from_segments is set to True while got num_test_clips: %d'
        % num_test_clips
    )

  # Parse frames.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenFeature((), dtype=tf.string),
        output_name=output_feature_name)
    sampler_builder.add_fn(
        fn=lambda x: tf.expand_dims(x, axis=0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_expand_dim')
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  if sample_around_keyframe:
    # Sample clip around keyframe.
    sample_around_keyframe_fn = functools.partial(
        processors.sample_sequence_around_keyframe,
        num_steps=num_frames,
        stride=temporal_stride,
        sample_target_key=output_feature_name)
    sampler_builder.add_fn(
        fn=sample_around_keyframe_fn,
        fn_name='{}_sample_around_keyframe'.format(output_feature_name))
  elif sample_from_segments:
    sample_segment_fn = functools.partial(
        processors.sample_sequence_by_segment,
        num_steps=num_frames,
        sample_target_key=output_feature_name,
        is_training=is_training)
    sampler_builder.add_fn(
        fn=sample_segment_fn,
        fn_name='{}_segment_sample'.format(output_feature_name))
  elif is_training:
    # Sample random clip.
    def sample_sequence_fn(x, state):
      return processors.sample_sequence(
          x,
          num_steps=num_frames, random=True, stride=temporal_stride, seed=seed,
          state=state)
    sampler_builder.add_fn(
        fn=sample_sequence_fn,
        feature_name=output_feature_name,
        fn_name='{}_random_sample'.format(output_feature_name),
        stateful=sync_random_state)
  else:
    if num_test_clips > 1:
      sample_linespace_sequence_fn = functools.partial(
          processors.sample_linsapce_sequence,
          num_windows=num_test_clips,
          num_steps=num_frames,
          stride=temporal_stride)
      # Sample linspace clips.
      sampler_builder.add_fn(
          fn=sample_linespace_sequence_fn,
          feature_name=output_feature_name,
          fn_name='{}_linspace_sample'.format(output_feature_name))
    else:
      sample_sequence_fn = functools.partial(
          processors.sample_sequence,
          num_steps=num_frames, random=False, stride=temporal_stride, seed=None)
      # Sample middle clip.
      sampler_builder.add_fn(
          fn=sample_sequence_fn,
          feature_name=output_feature_name,
          fn_name='{}_middle_sample'.format(output_feature_name))

  # Decode JPEG string to tf.uint8.
  num_raw_channels = 3
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name='{}_decode_jpeg'.format(output_feature_name))

  # Image crop, resize or pad.
  if is_training:
    if augmentation_type.lower() == 'inception':
      min_aspect_ratio = augmentation_params['min_aspect_ratio']
      max_aspect_ratio = augmentation_params['max_aspect_ratio']
      min_area_ratio = augmentation_params['min_area_ratio']
      max_area_ratio = augmentation_params['max_area_ratio']
      # Inception-style image crop: random crop -> resize.
      def random_crop_resize_fn(x, state=None):
        return processors.random_crop_resize(
            x, output_height=crop_size, output_width=crop_size,
            num_frames=num_frames, num_channels=num_raw_channels,
            aspect_ratio=(min_aspect_ratio, max_aspect_ratio),
            area_range=(min_area_ratio, max_area_ratio),
            state=state)
      preprocessor_builder.add_fn(
          fn=random_crop_resize_fn,
          feature_name=output_feature_name,
          fn_name='{}_random_crop_resize'.format(output_feature_name),
          stateful=sync_random_state)
    elif augmentation_type.lower() == 'vgg':
      # VGG-style image crop: resize -> random crop.
      def resize_and_crop_fn(x, state):
        return processors.resize_and_crop(
            x,
            min_resize=min_resize,
            crop_size=crop_size, is_flow=False, is_random=True,
            state=state)
      preprocessor_builder.add_fn(
          fn=resize_and_crop_fn,
          feature_name=output_feature_name,
          fn_name='{}_resize_random_crop'.format(output_feature_name),
          stateful=sync_random_state)
    elif augmentation_type.lower() == 'ava':
      # AVA-style image aug: random_crop -> resize -> random pad.
      def random_square_crop_by_scale_fn(x, state=None):
        return processors.random_square_crop_by_scale(
            image=x,
            scale_min=augmentation_params['scale_min'],
            scale_max=augmentation_params['scale_max'],
            state=state)
      preprocessor_builder.add_fn(
          fn=random_square_crop_by_scale_fn,
          feature_name=output_feature_name,
          fn_name='{}_random_square_crop_by_scale'.format(output_feature_name),
          stateful=sync_random_state)
      def resize_and_pad_fn(x, state=None):
        return processors.resize_and_pad(
            frames=x,
            max_resize=crop_size,
            pad_size=crop_size,
            random=True,
            state=state)
      preprocessor_builder.add_fn(
          fn=resize_and_pad_fn,
          feature_name=output_feature_name,
          fn_name='{}_resize_random_pad'.format(output_feature_name),
          # Use state to keep coherence between modalities if requested.
          stateful=sync_random_state)
    else:
      raise ValueError('Unrecognized augmentation_type: %s' %
                       augmentation_type)

    if random_flip_image:
      def random_flip_left_right_fn(x, state=None):
        return processors.random_flip_left_right(
            x, seed=seed, is_flow=False, state=state)
      preprocessor_builder.add_fn(
          fn=random_flip_left_right_fn,
          feature_name=output_feature_name,
          fn_name='{}_random_flip'.format(output_feature_name),
          stateful=sync_random_state)
  else:
    # Crop images, either a 3-view crop or a central crop.
    if multi_crop:
      resize_smallest_fn = functools.partial(
          processors.resize_smallest,
          min_resize=min_resize,
          is_flow=False)
      # Resize images (resize happens only if necessary to save compute).
      preprocessor_builder.add_fn(
          fn=resize_smallest_fn,
          feature_name=output_feature_name,
          fn_name='{}_resize_smallest'.format(output_feature_name))
      # Multi crop of the frames.
      preprocessor_builder.add_fn(
          fn=lambda x: processors.multi_crop_image(x, crop_size, crop_size),
          feature_name=output_feature_name,
          fn_name='{}_multi_crop'.format(output_feature_name))
    else:
      if augmentation_type.lower() == 'ava':
        def resize_and_pad_fn(x, state=None):
          return processors.resize_and_pad(
              frames=x,
              max_resize=crop_size,
              pad_size=crop_size,
              random=False,
              state=state)
        preprocessor_builder.add_fn(
            fn=resize_and_pad_fn,
            feature_name=output_feature_name,
            fn_name='{}_resize_central_pad'.format(output_feature_name),
            stateful=sync_random_state)
      else:
        def resize_and_crop_fn(x, state=None):
          return processors.resize_and_crop(
              x,
              min_resize=min_resize,
              crop_size=crop_size,
              is_flow=False,
              is_random=False,
              state=state)
        preprocessor_builder.add_fn(
            fn=resize_and_crop_fn,
            feature_name=output_feature_name,
            fn_name='{}_resize_central_crop'.format(output_feature_name),
            stateful=sync_random_state)

  # Apply extra augmentation policy.
  if is_training:
    if randaug_params is not None and autoaug_params is not None:
      raise ValueError('Choose to apply one of data augmentation policies: '
                       'randaug and autoaug.')

    if autoaug_params is not None:
      augmenter = augment.AutoAugment(
          augmentation_name=autoaug_params['augmentation_name'],
          cutout_const=autoaug_params['cutout_const'],
          translate_const=autoaug_params['translate_const'])
      preprocessor_builder.add_fn(
          fn=augmenter.distort,
          feature_name=output_feature_name,
          fn_name='{}_autoaug'.format(output_feature_name))

    if randaug_params is not None:
      augmenter = augment.RandAugment(
          num_layers=randaug_params['num_layers'],
          magnitude=randaug_params['magnitude'],
          cutout_const=randaug_params['cutout_const'],
          translate_const=randaug_params['translate_const'],
          prob_to_apply=randaug_params['prob_to_apply'],
          exclude_ops=randaug_params['exclude_ops'])
      preprocessor_builder.add_fn(
          fn=augmenter.distort,
          feature_name=output_feature_name,
          fn_name='{}_randaug'.format(output_feature_name))

  # Cast the frames in float32, normalizing according to zero_centering_image.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.normalize_image(x, zero_centering_image),
      feature_name=output_feature_name,
      fn_name='{}_normalize'.format(output_feature_name))

  if (num_test_clips > 1 or multi_crop) and not is_training:
    # In this case, multiple clips are merged together in batch dimenstion which
    # will be B * num_test_clips.
    def reshape_fn(x):
      target_shape = (-1, num_frames, x.shape[-3], x.shape[-2], x.shape[-1])
      return tf.reshape(x, target_shape)
    postprocessor_builder.add_fn(
        fn=reshape_fn,
        feature_name=output_feature_name,
        fn_name='{}_reshape'.format(output_feature_name))


def add_context_label(
    parser_builder: builders.SequenceExampleParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_label_index_feature_name: str = 'clip/key_frame/bbox/label/index',
    output_label_index_feature_name: str = builders.LABEL_INDEX_FEATURE_NAME,
    input_label_name_feature_name: str = 'clip/key_frame/bbox/label/string',
    output_label_name_feature_name: str = builders.LABEL_NAME_FEATURE_NAME,
    # Label related parameters.
    num_frames: int = 1,
    num_instances_per_frame: int = 5,
    zero_based_index: bool = False,
    one_hot_label: bool = True,
    num_classes: Optional[int] = None,
    add_label_name: bool = False):
  """Adds functions to process label feature to builders.

  Args:
    parser_builder: An instance of a builders.SequenceExampleParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_label_index_feature_name: Name of the label index feature in the input
      SequenceExample. Exposing this as an argument allows using this function
      for different label features.
    output_label_index_feature_name: Name of the label index feature in the
      output features dictionary. Exposing this as an argument allows using this
      function for different label features.
    input_label_name_feature_name: Name of the label name feature in the input
      SequenceExample. Exposing this as an argument allows using this function
      for different label features.
    output_label_name_feature_name: Name of the label name feature in the
      output features dictionary. Exposing this as an argument allows using this
      function for different label features.
    num_frames: The number of frames. If the num_frames > 1, the labels will be
      duplicated.
    num_instances_per_frame: The number of label instances per frames.
    zero_based_index: Whether the raw index are zero based. If not, converted to
      the zero based index as the output.
    one_hot_label: Return labels as one hot tensors. If is_multi_label is True,
      one hot tensor might have multiple ones.
    num_classes: Total number of classes in the dataset. It has to be procided
      if one_hot_label is True.
    add_label_name: Also return the name of the label. Not yet supported for
      multi label.
  """
  # Validate parameters.
  if one_hot_label and not num_classes:
    raise ValueError(
        'num_classes should be given when requesting one hot label.')

  # Parse label.
  parser_builder.parse_feature(
      feature_name=input_label_index_feature_name,
      feature_type=tf.io.VarLenFeature(dtype=tf.int64),
      output_name=output_label_index_feature_name,
      is_context=True)

  # Densify labels tensor in order to support multi label case.
  sampler_builder.add_fn(
      fn=lambda x: tf.sparse.to_dense(x, default_value=-1),
      feature_name=output_label_index_feature_name,
      fn_name='{}_sparse_to_dense'.format(output_label_index_feature_name))

  # Crop or pad labels to max_num_instance.
  crop_or_pad_features_fn = functools.partial(
      processors.crop_or_pad_features,
      max_num_features=num_instances_per_frame,
      feature_dimension=1,
      constant_values=-1)
  preprocessor_builder.add_fn(
      fn=crop_or_pad_features_fn,
      feature_name=output_label_index_feature_name,
      fn_name='{}_crop_or_pad'.format(output_label_index_feature_name))

  if num_frames > 1:
    preprocessor_builder.add_fn(
        fn=lambda x: tf.tile(x, [num_frames, 1]),
        feature_name=output_label_index_feature_name,
        fn_name='{}_duplicate'.format(output_label_index_feature_name))

  # Reshape the feature vector in [T, N].
  target_shape = [num_frames, num_instances_per_frame]
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, target_shape),
      feature_name=output_label_index_feature_name,
      fn_name='{}_reshape'.format(output_label_index_feature_name))

  # Convert the label id to be zero-indexed.
  if not zero_based_index:
    preprocessor_builder.add_fn(
        fn=lambda x: x - 1,
        feature_name=output_label_index_feature_name,
        fn_name='{}_zero_based'.format(output_label_index_feature_name))

  # Replace label index by one hot representation.
  if one_hot_label:
    preprocessor_builder.add_fn(
        fn=lambda x: tf.one_hot(x, num_classes),
        feature_name=output_label_index_feature_name,
        fn_name='{}_one_hot'.format(output_label_index_feature_name))

  if add_label_name:
    parser_builder.parse_feature(
        feature_name=input_label_name_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_label_name_feature_name,
        is_context=True)
    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name=output_label_name_feature_name,
        fn_name='{}_sparse_to_dense'.format(output_label_name_feature_name))

    # Crop or pad labels to max_num_instance.
    crop_or_pad_features_fn = functools.partial(
        processors.crop_or_pad_features,
        max_num_features=num_instances_per_frame,
        feature_dimension=1,
        constant_values='')
    preprocessor_builder.add_fn(
        fn=crop_or_pad_features_fn,
        feature_name=output_label_name_feature_name,
        fn_name='{}_crop_or_pad'.format(output_label_name_feature_name))

    if num_frames > 1:
      preprocessor_builder.add_fn(
          fn=lambda x: tf.tile(x, [num_frames, 1]),
          feature_name=output_label_name_feature_name,
          fn_name='{}_duplicate'.format(output_label_name_feature_name))

    # Reshape the feature vector in [T, N].
    target_shape = [num_frames, num_instances_per_frame]
    preprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(x, target_shape),
        feature_name=output_label_name_feature_name,
        fn_name='{}_reshape'.format(output_label_name_feature_name))


def add_frame_label(
    parser_builder: builders.SequenceExampleParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_label_name: str = 'region/bbox/xmin',
    output_label_name: str = 'instance_xmin',
    dtype: tf.dtypes.DType = tf.float32,
    # Instance related parameters.
    is_training: bool = True,
    num_instances_per_frame: int = 5,
    num_frames: int = 32,
    temporal_stride: int = 1,
    sync_random_state: bool = True,
    seed: Optional[int] = None):
  """Adds functions to process label feature to builders.

  Args:
    parser_builder: An instance of a builders.SequenceExampleParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_label_name: The per frame label key stored in the tfse for parsing.
    output_label_name: The output feature name. Exposing this as an argument
      allows using this function for
    dtype: The data type to be parsed.
    is_training: Whether in training mode.
    num_instances_per_frame: The number of instances label per frame.
    num_frames: The number of frames in a video clip.
    temporal_stride: The temporal sample stride.
    sync_random_state: Whether to sync random state between features.
    seed: the random seed.
  """
  # Parse per-frame label.
  parser_builder.parse_feature(
      feature_name=input_label_name,
      # Entire signal stored in one Feature.
      feature_type=tf.io.VarLenFeature(dtype=dtype),
      output_name=output_label_name)

  sampler_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_label_name,
      fn_name='{}_sparse_to_dense'.format(output_label_name))

  # Temporal sampler.
  if is_training:
    def sample_sequence_fn(x, state=None):
      return processors.sample_sequence(
          x,
          num_steps=num_frames,
          random=True,
          stride=temporal_stride,
          seed=seed,
          state=state)
    # Sample random clip.
    sampler_builder.add_fn(
        fn=sample_sequence_fn,
        feature_name=output_label_name,
        fn_name='{}_random_sample'.format(output_label_name),
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    sample_sequence_fn = functools.partial(
        processors.sample_sequence,
        num_steps=num_frames,
        random=False,
        stride=temporal_stride,
        seed=None)
    # Sample middle clip.
    sampler_builder.add_fn(
        fn=sample_sequence_fn,
        feature_name=output_label_name,
        fn_name='{}_middle_sample'.format(output_label_name))

  # Crop or pad labels to num_instances_per_frame.
  crop_or_pad_features_fn = functools.partial(
      processors.crop_or_pad_features,
      max_num_features=num_instances_per_frame,
      feature_dimension=1,
      constant_values=-1)
  preprocessor_builder.add_fn(
      fn=crop_or_pad_features_fn,
      feature_name=output_label_name,
      fn_name='{}_crop_or_pad'.format(output_label_name))

  # Reshape the feature vector in [T, N].
  target_shape = [num_frames, num_instances_per_frame]
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, target_shape),
      feature_name=output_label_name,
      fn_name='{}_reshape'.format(output_label_name))


def merge_multi_labels(
    inputs: MutableMapping[str, tf.Tensor],
    num_classes: int,
    input_label_name: str = 'label',
    input_boxes_name: str = 'instances_position',
    input_masks_name: str = 'instances_mask',
    output_nonmerge_boxes_name: str = 'nonmerge_instances_position',
    output_nonmerge_label_name: str = 'nonmerge_label',
    quantization_bins: int = 1000):
  """Merges boxes with the same coordinates and returns k-hot labels.

  Args:
    inputs: An inputs dictionary. Containing at least the following fields:
      * instances_position:  A tf.float32 tensor with shape
        [num_frames, num_boxes, 4] holding boxes. Only normalized coordinates in
        form [ymin, xmin, ymax, xmax] are allowed.
      * label: A tf.int32 tensor with shape [num_frames, num_boxes] holding
          zero-indexed classes. -1 means that label is padded and thus invalid.
      * instances_mask: A tf.bool tensor with shape [num_frames, num_boxes]
        holding class confidences.
    num_classes: The maximum number of classes for the dataset.
    input_label_name: The input label tensor name. The label should be the
      0-indexed, unique label per box.
    input_boxes_name: The input box tensor name.
    input_masks_name: The input binary mask tensor name.
    output_nonmerge_boxes_name: The output box name holding the none-merged
      might be duplicated boxes.
    output_nonmerge_label_name: The output label index per box.
    quantization_bins: The number of bins used to quantize the box coordinate.

  Returns:
    The output dictionary contains the merged multilabel labels/boxes and
      none-merged labels/boxes pairs.
  """
  unique_labels = inputs[input_label_name]
  unique_boxes = inputs[input_boxes_name]
  unique_weights = inputs[input_masks_name]

  if unique_labels.shape.rank != 2:
    raise ValueError('one_hot should be turned off if merge_multi_labels.')

  num_instances_per_frame = unique_boxes.shape.as_list()[-2]

  labels = tf.unstack(unique_labels, axis=0)
  boxes = tf.unstack(unique_boxes, axis=0)
  weights = tf.unstack(unique_weights, axis=0)

  merged_boxes = []
  merged_labels = []
  merged_weights = []

  def true_fn(box, label, weight,
              num_classes=num_classes,
              num_instances_per_frame=num_instances_per_frame,
              quantization_bins=quantization_bins):
    # The label is 0-index and the invalid labels are padded with -1. We create
    # a binary mask here to filter out the invalid/padded label.
    valid_mask = tf.greater(label, -1)

    confidence = tf.cast(weight, dtype=tf.float32)
    merged_box, merged_label, merged_confidence, _ = (
        merge_boxes_with_multiple_labels(
            tf.boolean_mask(box, valid_mask),
            tf.boolean_mask(label, valid_mask),
            tf.boolean_mask(confidence, valid_mask),
            num_classes=num_classes,
            quantization_bins=quantization_bins))

    merged_box = pad_or_clip_nd(
        merged_box, [num_instances_per_frame, 4])
    merged_label = pad_or_clip_nd(
        merged_label, [num_instances_per_frame, num_classes])
    merged_confidence = pad_or_clip_nd(
        merged_confidence, [num_instances_per_frame, num_classes])
    merged_label = tf.cast(merged_label, dtype=tf.float32)
    merged_weight = tf.cast(tf.reduce_max(merged_confidence, axis=-1), tf.bool)
    return merged_box, merged_label, merged_weight

  def false_fn(box, label, weight, num_classes=num_classes):
    label = tf.one_hot(label, num_classes)
    return box, label, weight

  for i in range(len(labels)):
    # determine whether this frame is the keyframe by examining the label id.
    # any instance with label id > -1 means such frame is labeled.
    contains_label = tf.greater(tf.reduce_max(labels[i]), -1)

    # pylint: disable=cell-var-from-loop
    merged_box, merged_label, merged_weight = tf.cond(
        contains_label,
        true_fn=lambda: true_fn(boxes[i], labels[i], weights[i]),
        false_fn=lambda: false_fn(boxes[i], labels[i], weights[i]))
    # pylint: enable=cell-var-from-loop

    merged_boxes.append(merged_box)
    merged_labels.append(merged_label)
    merged_weights.append(merged_weight)

  # All values to the input labels/boxes/masks will now store the merged value.
  inputs[input_label_name] = tf.stack(merged_labels, axis=0)
  inputs[input_boxes_name] = tf.stack(merged_boxes, axis=0)
  inputs[input_masks_name] = tf.stack(merged_weights, axis=0)

  # The difference between groundtruth classes/boxes and above is the
  # groundtruth boxes maybe duplicated since one box may have multiple labels.
  inputs[output_nonmerge_label_name] = unique_labels
  inputs[output_nonmerge_boxes_name] = unique_boxes
  return inputs


def add_context_box_dim(
    parser_builder: builders.SequenceExampleParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_box_dim_name: str = 'clip/key_frame/bbox/xmin',
    output_box_dim_name: str = 'keyframe_xmin',
    num_frames: int = 1,
    num_instances_per_frame: int = 5,
    default_value: float = -1):
  """Adds functions to process label feature to builders.

  Args:
    parser_builder: An instance of a builders.BaseParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_box_dim_name: The box dim key stored in the tfse for parsing.
    output_box_dim_name: The output feature name. Exposing this as an argument
      allows using this function for
    num_frames: The number of frames the boxes need to be duplicated. This
      parameter is added to adapt the format of boxes in this codebase.
    num_instances_per_frame: The number of instance per keyframe.
    default_value: The default value to pad to num_instances_per_frame.
  """
  # Parse box dim.
  parser_builder.parse_feature(
      feature_name=input_box_dim_name,
      # Entire signal stored in one Feature.
      feature_type=tf.io.VarLenFeature(dtype=tf.float32),
      output_name=output_box_dim_name,
      is_context=True)

  sampler_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_box_dim_name,
      fn_name='{}_sparse_to_dense'.format(output_box_dim_name))

  # Crop or pad boxes to num_instances_per_frame.
  crop_or_pad_features_fn = functools.partial(
      processors.crop_or_pad_features,
      max_num_features=num_instances_per_frame,
      feature_dimension=1,
      constant_values=default_value)
  preprocessor_builder.add_fn(
      fn=crop_or_pad_features_fn,
      feature_name=output_box_dim_name,
      fn_name='{}_crop_or_pad'.format(output_box_dim_name))

  # Duplicate keyframe boxes to all frames.
  if num_frames > 1:
    preprocessor_builder.add_fn(
        fn=lambda x: tf.tile(x, [num_frames, 1]),
        feature_name=output_box_dim_name,
        fn_name='{}_duplicate'.format(output_box_dim_name))

  # Reshape the feature vector in [T, N].
  target_shape = [num_frames, num_instances_per_frame]
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, target_shape),
      feature_name=output_box_dim_name,
      fn_name='{}_reshape'.format(output_box_dim_name))


def add_context_features(
    parser_builder: builders.SequenceExampleParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_feature_name: str = 'clip/p_scores',
    output_feature_name: str = 'p_scores',
    max_num_features: int = 64,
    feature_dimension: int = 35):
  """Adds functions to process context feature to builders.

  Args:
    parser_builder: An instance of a builders.BaseParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_feature_name: The box dim key stored in the tfse for parsing.
    output_feature_name: The output feature name. Exposing this as an argument
      allows using this function for
    max_num_features: The number of features to be processed.
    feature_dimension: The feature dimension.
  """
  # Parse box dim.
  parser_builder.parse_feature(
      feature_name=input_feature_name,
      # Entire signal stored in one Feature.
      feature_type=tf.io.VarLenFeature(dtype=tf.float32),
      output_name=output_feature_name,
      is_context=True)

  sampler_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_feature_name,
      fn_name='{}_sparse_to_dense'.format(output_feature_name))

  # Crop or pad boxes to num_instances_per_frame.
  crop_or_pad_features_fn = functools.partial(
      processors.crop_or_pad_features,
      max_num_features=max_num_features,
      feature_dimension=feature_dimension,
      constant_values=-1)
  preprocessor_builder.add_fn(
      fn=crop_or_pad_features_fn,
      feature_name=output_feature_name,
      fn_name='{}_crop_or_pad'.format(output_feature_name))

  # Reshape the feature vector in [T, N].
  target_shape = [max_num_features, feature_dimension]
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, target_shape),
      feature_name=output_feature_name,
      fn_name='{}_reshape'.format(output_feature_name))


def add_instance_box_dim(
    parser_builder: builders.SequenceExampleParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_box_dim_name: str = 'region/bbox/xmin',
    output_box_dim_name: str = 'instance_xmin',
    # Instance related parameters.
    sample_around_keyframe: bool = False,
    sample_random: bool = True,
    num_instances_per_frame: int = 5,
    num_frames: int = 32,
    temporal_stride: int = 1,
    sync_random_state: bool = True):
  """Adds functions to process label feature to builders.

  Args:
    parser_builder: An instance of a builders.BaseParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_box_dim_name: The input feature name.
    output_box_dim_name: The output feature name.
    sample_around_keyframe: Whether to sample the sequence around the keyframe.
      If True, it requires keyframe id is known.
    sample_random:  Whether to perform random sampling.
    num_instances_per_frame: The number of instances per frame.
    num_frames: The number of frames to sample.
    temporal_stride: The temopral sampling stride.
    sync_random_state: Whether to sync the random state.
  """
  if sample_random and sample_around_keyframe:
    raise ValueError(
        'sample_random and sample_around_keyframe cannot be both True.')

  # Parse box dim.
  parser_builder.parse_feature(
      feature_name=input_box_dim_name,
      # Entire signal stored in one Feature.
      feature_type=tf.io.VarLenFeature(dtype=tf.float32),
      output_name=output_box_dim_name)

  sampler_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_box_dim_name,
      fn_name='{}_sparse_to_dense'.format(output_box_dim_name))

  # Temporal sampler.
  if sample_random:
    # Sample random clip.
    def sample_sequence_fn(x, state=None):
      return processors.sample_sequence(
          sequence=x,
          num_steps=num_frames,
          random=True,
          stride=temporal_stride,
          state=state)
    sampler_builder.add_fn(
        fn=sample_sequence_fn,
        feature_name=output_box_dim_name,
        fn_name='{}_random_sample'.format(output_box_dim_name),
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  elif sample_around_keyframe:
    sample_around_keyframe_fn = functools.partial(
        processors.sample_sequence_around_keyframe,
        num_steps=num_frames,
        stride=temporal_stride,
        sample_target_key=output_box_dim_name)
    sampler_builder.add_fn(
        fn=sample_around_keyframe_fn,
        fn_name='{}_sample_around_keyframe'.format(output_box_dim_name))
  else:
    # Sample middle clip.
    sample_sequence_fn = functools.partial(
        processors.sample_sequence,
        num_steps=num_frames,
        random=False,
        stride=temporal_stride)
    sampler_builder.add_fn(
        fn=sample_sequence_fn,
        feature_name=output_box_dim_name,
        fn_name='{}_middle_sample'.format(output_box_dim_name))

  # Crop or pad boxes to num_instances_per_frame.
  crop_or_pad_features_fn = functools.partial(
      processors.crop_or_pad_features,
      max_num_features=num_instances_per_frame,
      feature_dimension=1,
      constant_values=-1)
  preprocessor_builder.add_fn(
      fn=crop_or_pad_features_fn,
      feature_name=output_box_dim_name,
      fn_name='{}_crop_or_pad'.format(output_box_dim_name))

  # Reshape the feature vector in [T, N, C].
  target_shape = [num_frames, num_instances_per_frame]
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, target_shape),
      feature_name=output_box_dim_name,
      fn_name='{}_reshape'.format(output_box_dim_name))


def group_instance_box_dims(
    preprocessor_builder: builders.PreprocessorBuilder,
    output_position_name: str = 'instances_position',
    box_key_prefix: str = 'instance'):
  """Groups instance box dims into a compact feature vector.

  Args:
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    output_position_name: The output position tensor name.
    box_key_prefix: The box key prefix.
  """

  def _group_box_dims_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Utility function to group box dimensions."""
    box_dims = [
        inputs.pop('{}_ymin'.format(box_key_prefix)),
        inputs.pop('{}_xmin'.format(box_key_prefix)),
        inputs.pop('{}_ymax'.format(box_key_prefix)),
        inputs.pop('{}_xmax'.format(box_key_prefix)),
    ]

    inputs[output_position_name] = tf.stack(
        box_dims, axis=-1, name='stack_{}'.format(output_position_name))
    return inputs

  preprocessor_builder.add_fn(
      fn=_group_box_dims_fn,
      fn_name='{}_stack'.format(output_position_name))


def infer_instances_mask_from_position(
    inputs: MutableMapping[str, tf.Tensor],
    instances_position_name: str = 'instances_position'):
  """Infers binary instances mask from instance positions."""
  instances_position = inputs[instances_position_name]
  instances_mask = tf.reduce_sum(instances_position, axis=-1) > 0
  inputs['instances_mask'] = instances_mask
  return inputs


def filter_instances_box_by_score(
    inputs: MutableMapping[str, tf.Tensor],
    box_key_prefix: str = 'detected_instances',
    score_threshold: float = 0.2):
  """Filters the low-score detected instance.

  Args:
    inputs: The input dictionary contains boxes and masks.
    box_key_prefix: The instances position key prefix.
    score_threshold: The detection score to threshold.

  Returns:
    The output dictionary contains filtered boxes and mask.
  """
  instances_position = inputs.pop(f'{box_key_prefix}_position')
  instances_score = inputs[f'{box_key_prefix}_score']
  num_frames, num_instances = instances_score.shape.as_list()

  masks_list = tf.unstack(instances_score > score_threshold, num_frames, axis=0)
  position_list = tf.unstack(instances_position, num_frames, axis=0)
  constant_padding = tf.ones_like(position_list[0], dtype=tf.float32) * -1.
  instances_position_filtered = []
  for mask, position in zip(masks_list, position_list):
    position_filtered = tf.boolean_mask(position, mask)
    position_filtered = tf.concat([position_filtered, constant_padding], axis=0)
    position_filtered = tf.slice(position_filtered, [0, 0], [num_instances, -1])
    instances_position_filtered.append(position_filtered)
  instances_position_filtered = tf.stack(instances_position_filtered, axis=0)

  inputs[f'{box_key_prefix}_position'] = instances_position_filtered
  inputs[f'{box_key_prefix}_mask'] = tf.reduce_sum(
      instances_position_filtered, axis=-1) > 0
  return inputs


def adjust_positions(preprocessor_builder: builders.PreprocessorBuilder,
                     input_tensor_name: str = 'instances_position',
                     output_tensor_name: str = 'instances_position'):
  """Adjusts box positions based on image/video data augmentation.

  Args:
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_tensor_name: The name of input tensor.
    output_tensor_name: The name of output tensor.
  """

  def _resize_and_crop(boxes, image_info, is_flip=None):
    """Resizes and crop boxes."""
    # input boxes in shape [T, N, 4]
    # image_info in shape [4, 2]
    original_image_size, target_image_size, scale, offset = tf.unstack(
        image_info, axis=0)
    # 1) Un-normalize boxes to absolute coordinate.
    boxes = box_ops.denormalize_boxes(boxes, original_image_size)
    # 2) Adjusts box coordinates based on image_scale and offset.
    boxes *= tf.tile(scale[None, None, :], [1, 1, 2])
    boxes -= tf.tile(offset[None, None, :], [1, 1, 2])
    # 3) Clips the boxes.
    boxes = box_ops.clip_boxes(boxes, target_image_size)
    # 4) Normalize boxes.
    boxes = box_ops.normalize_boxes(boxes, target_image_size)
    # 5) Random flip boxes.
    if is_flip is not None:
      boxes = tf.cond(
          tf.equal(is_flip, 1),
          lambda: box_ops.horizontal_flip_boxes(boxes),
          lambda: boxes)
    return boxes

  def _adjust_boxes_fn(boxes: tf.Tensor,
                       state: MutableMapping[str, Any]) -> tf.Tensor:
    return _resize_and_crop(
        boxes,
        image_info=state['image_info'],
        is_flip=state.get('flip_left_right_is_flipped', None))

  preprocessor_builder.add_fn(
      fn=_adjust_boxes_fn,
      feature_name=input_tensor_name,
      fn_name='{}_adjust'.format(output_tensor_name),
      stateful=True)


def update_valid_instances_mask(
    inputs: Dict[str, tf.Tensor],
    instances_position_name: str = 'instances_position',
    instances_mask_name: str = 'instances_mask') -> Dict[str, tf.Tensor]:
  """Filters empty boxes and mark them in instances_mask.

  Args:
    inputs: The dictionary contains the input tensors.
    instances_position_name: The box name stored in the dictionary.
    instances_mask_name: The binary instance mask key in the inputs.

  Returns:
    The input dictionary with the updated instances mask.
  """
  boxes = inputs[instances_position_name]
  height = boxes[..., 2] - boxes[..., 0]
  width = boxes[..., 3] - boxes[..., 1]
  instances_mask = tf.logical_and(tf.greater(height, 0), tf.greater(width, 0))
  if instances_mask_name in inputs:
    instances_mask = tf.logical_and(inputs[instances_mask_name], instances_mask)
  inputs[instances_mask_name] = instances_mask
  return inputs


def apply_default_color_augmentations(
    preprocessor_builder: builders.PreprocessorBuilder,
    image_feature_name: str = builders.IMAGE_FEATURE_NAME,
    zero_centering_image: bool = False):
  """Applies default color augmentations on images.

  Args:
    preprocessor_builder: The prerpocessor builder.
    image_feature_name: The image feature name.
    zero_centering_image: Whether the image has been zero centered.
  """
  preprocessor_builder.add_fn(
      functools.partial(
          processors.random_color_augmentation,
          zero_centering_image=zero_centering_image),
      feature_name=image_feature_name,
      fn_name='random_jitter_color')
  preprocessor_builder.add_fn(
      functools.partial(
          processors.random_blur_and_solarize,
          zero_centering_image=zero_centering_image),
      feature_name=image_feature_name,
      fn_name='random_blur_and_solarize')


def get_shards(table_name: str) -> list[str]:
  """Expands table into a list of sharded filenames."""
  filenames = []
  if '@' in table_name:
    base_filename, num_shards = table_name.split('@')
    num_shards = int(num_shards)
    for ind in range(num_shards):
      filename = '{}.{:05d}-of-{:05d}'.format(base_filename, ind, num_shards)
      filenames.append(filename)
  else:
    filenames.append(table_name)
  return filenames
