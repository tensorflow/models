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

"""Wrapper for providing semantic segmentation video data."""

import tensorflow as tf
from feelvos import input_preprocess
from feelvos import model
from feelvos.utils import mask_damaging
from feelvos.utils import train_utils

slim = tf.contrib.slim
dataset_data_provider = slim.dataset_data_provider


MIN_LABEL_COUNT = 10


def decode_image_sequence(tensor, image_format='jpeg', shape=None,
                          channels=3, raw_dtype=tf.uint8):
  """Decodes a sequence of images.

  Args:
    tensor: the tensor of strings to decode, shape: [num_images]
    image_format: a string (possibly tensor) with the format of the image.
      Options include 'jpeg', 'png', and 'raw'.
    shape: a list or tensor of the decoded image shape for a single image.
    channels: if 'shape' is None, the third dimension of the image is set to
      this value.
    raw_dtype: if the image is encoded as raw bytes, this is the method of
      decoding the bytes into values.
  Returns:
    The decoded images with shape [time, height, width, channels].
  """
  handler = slim.tfexample_decoder.Image(
      shape=shape, channels=channels, dtype=raw_dtype, repeated=True)
  return handler.tensors_to_item({'image/encoded': tensor,
                                  'image/format': image_format})


def _get_data(data_provider, dataset_split, video_frames_are_decoded):
  """Gets data from data provider.

  Args:
    data_provider: An object of slim.data_provider.
    dataset_split: Dataset split.
    video_frames_are_decoded: Boolean, whether the video frames are already
        decoded

  Returns:
    image: Image Tensor.
    label: Label Tensor storing segmentation annotations.
    object_label: An integer refers to object_label according to labelmap. If
      the example has more than one object_label, take the first one.
    image_name: Image name.
    height: Image height.
    width: Image width.
    video_id: String tensor representing the name of the video.

  Raises:
    ValueError: Failed to find label.
  """

  if video_frames_are_decoded:
    image, = data_provider.get(['image'])
  else:
    image, = data_provider.get(['image/encoded'])

  # Some datasets do not contain image_name.
  if 'image_name' in data_provider.list_items():
    image_name, = data_provider.get(['image_name'])
  else:
    image_name = tf.constant('')

  height, width = data_provider.get(['height', 'width'])

  label = None
  if dataset_split != 'test':
    if video_frames_are_decoded:
      if 'labels_class' not in data_provider.list_items():
        raise ValueError('Failed to find labels.')
      label, = data_provider.get(['labels_class'])
    else:
      key = 'segmentation/object/encoded'
      if key not in data_provider.list_items():
        raise ValueError('Failed to find labels.')
      label, = data_provider.get([key])

  object_label = None
  video_id, = data_provider.get(['video_id'])

  return image, label, object_label, image_name, height, width, video_id


def _has_foreground_and_background_in_first_frame(label, subsampling_factor):
  """Checks if the labels have foreground and background in the first frame.

  Args:
    label: Label tensor of shape [num_frames, height, width, 1].
    subsampling_factor: Integer, the subsampling factor.

  Returns:
    Boolean, whether the labels have foreground and background in the first
      frame.
  """
  h, w = train_utils.resolve_shape(label)[1:3]
  label_downscaled = tf.squeeze(
      tf.image.resize_nearest_neighbor(label[0, tf.newaxis],
                                       [h // subsampling_factor,
                                        w // subsampling_factor],
                                       align_corners=True),
      axis=0)
  is_bg = tf.equal(label_downscaled, 0)
  is_fg = tf.logical_not(is_bg)
  # Just using reduce_any was not robust enough, so lets make sure the count
  # is above MIN_LABEL_COUNT.
  fg_count = tf.reduce_sum(tf.cast(is_fg, tf.int32))
  bg_count = tf.reduce_sum(tf.cast(is_bg, tf.int32))
  has_bg = tf.greater_equal(fg_count, MIN_LABEL_COUNT)
  has_fg = tf.greater_equal(bg_count, MIN_LABEL_COUNT)
  return tf.logical_and(has_bg, has_fg)


def _has_foreground_and_background_in_first_frame_2(label,
                                                    decoder_output_stride):
  """Checks if the labels have foreground and background in the first frame.

  Second attempt, this time we use the actual output dimension for resizing.

  Args:
    label: Label tensor of shape [num_frames, height, width, 1].
    decoder_output_stride: Integer, the stride of the decoder output.

  Returns:
    Boolean, whether the labels have foreground and background in the first
      frame.
  """
  h, w = train_utils.resolve_shape(label)[1:3]
  h_sub = model.scale_dimension(h, 1.0 / decoder_output_stride)
  w_sub = model.scale_dimension(w, 1.0 / decoder_output_stride)
  label_downscaled = tf.squeeze(
      tf.image.resize_nearest_neighbor(label[0, tf.newaxis], [h_sub, w_sub],
                                       align_corners=True), axis=0)
  is_bg = tf.equal(label_downscaled, 0)
  is_fg = tf.logical_not(is_bg)
  # Just using reduce_any was not robust enough, so lets make sure the count
  # is above MIN_LABEL_COUNT.
  fg_count = tf.reduce_sum(tf.cast(is_fg, tf.int32))
  bg_count = tf.reduce_sum(tf.cast(is_bg, tf.int32))
  has_bg = tf.greater_equal(fg_count, MIN_LABEL_COUNT)
  has_fg = tf.greater_equal(bg_count, MIN_LABEL_COUNT)
  return tf.logical_and(has_bg, has_fg)


def _has_enough_pixels_of_each_object_in_first_frame(
    label, decoder_output_stride):
  """Checks if for each object (incl. background) enough pixels are visible.

  During test time, we will usually not see a reference frame in which only
  very few pixels of one object are visible. These cases can be problematic
  during training, especially if more than the 1-nearest neighbor is used.
  That's why this function can be used to detect and filter these cases.

  Args:
    label: Label tensor of shape [num_frames, height, width, 1].
    decoder_output_stride: Integer, the stride of the decoder output.

  Returns:
    Boolean, whether the labels have enough pixels of each object in the first
      frame.
  """
  h, w = train_utils.resolve_shape(label)[1:3]
  h_sub = model.scale_dimension(h, 1.0 / decoder_output_stride)
  w_sub = model.scale_dimension(w, 1.0 / decoder_output_stride)
  label_downscaled = tf.squeeze(
      tf.image.resize_nearest_neighbor(label[0, tf.newaxis], [h_sub, w_sub],
                                       align_corners=True), axis=0)
  _, _, counts = tf.unique_with_counts(
      tf.reshape(label_downscaled, [-1]))
  has_enough_pixels_per_object = tf.reduce_all(
      tf.greater_equal(counts, MIN_LABEL_COUNT))
  return has_enough_pixels_per_object


def get(dataset,
        num_frames_per_video,
        crop_size,
        batch_size,
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        preprocess_image_and_label=True,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None,
        batch_capacity_factor=32,
        video_frames_are_decoded=False,
        decoder_output_stride=None,
        first_frame_finetuning=False,
        sample_only_first_frame_for_finetuning=False,
        sample_adjacent_and_consistent_query_frames=False,
        remap_labels_to_reference_frame=True,
        generate_prev_frame_mask_by_mask_damaging=False,
        three_frame_dataset=False,
        add_prev_frame_label=True):
  """Gets the dataset split for semantic segmentation.

  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.

  Args:
    dataset: An instance of slim Dataset.
    num_frames_per_video: The number of frames used per video
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
    preprocess_image_and_label: Boolean variable specifies if preprocessing of
      image and label will be performed or not.
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    dataset_split: Dataset split.
    is_training: Is training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.
    batch_capacity_factor: Batch capacity factor affecting the training queue
      batch capacity.
    video_frames_are_decoded: Boolean, whether the video frames are already
        decoded
    decoder_output_stride: Integer, the stride of the decoder output.
    first_frame_finetuning: Boolean, whether to only sample the first frame
      for fine-tuning.
    sample_only_first_frame_for_finetuning: Boolean, whether to only sample the
      first frame during fine-tuning. This should be False when using lucid or
      wonderland data, but true when fine-tuning on the first frame only.
      Only has an effect if first_frame_finetuning is True.
    sample_adjacent_and_consistent_query_frames: Boolean, if true, the query
      frames (all but the first frame which is the reference frame) will be
      sampled such that they are adjacent video frames and have the same
      crop coordinates and flip augmentation.
    remap_labels_to_reference_frame: Boolean, whether to remap the labels of
      the query frames to match the labels of the (downscaled) reference frame.
      If a query frame contains a label which is not present in the reference,
      it will be mapped to background.
    generate_prev_frame_mask_by_mask_damaging: Boolean, whether to generate
      the masks used as guidance from the previous frame by damaging the
      ground truth mask.
    three_frame_dataset: Boolean, whether the dataset has exactly three frames
      per video of which the first is to be used as reference and the two
      others are consecutive frames to be used as query frames.
    add_prev_frame_label: Boolean, whether to sample one more frame before the
      first query frame to obtain a previous frame label. Only has an effect,
      if sample_adjacent_and_consistent_query_frames is True and
      generate_prev_frame_mask_by_mask_damaging is False.

  Returns:
    A dictionary of batched Tensors for semantic segmentation.

  Raises:
    ValueError: dataset_split is None, or Failed to find labels.
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
  image, label, object_label, image_name, height, width, video_id = _get_data(
      data_provider, dataset_split, video_frames_are_decoded)

  sampling_is_valid = tf.constant(True)
  if num_frames_per_video is not None:
    total_num_frames = tf.shape(image)[0]
    if first_frame_finetuning or three_frame_dataset:
      if sample_only_first_frame_for_finetuning:
        assert not sample_adjacent_and_consistent_query_frames, (
            'this option does not make sense for sampling only first frame.')
        # Sample the first frame num_frames_per_video times.
        sel_indices = tf.tile(tf.constant(0, dtype=tf.int32)[tf.newaxis],
                              multiples=[num_frames_per_video])
      else:
        if sample_adjacent_and_consistent_query_frames:
          if add_prev_frame_label:
            num_frames_per_video += 1
          # Since this is first frame fine-tuning, we'll for now assume that
          # each sequence has exactly 3 images: the ref frame and 2 adjacent
          # query frames.
          assert num_frames_per_video == 3
          with tf.control_dependencies([tf.assert_equal(total_num_frames, 3)]):
            sel_indices = tf.constant([1, 2], dtype=tf.int32)
        else:
          # Sample num_frames_per_video - 1 query frames which are not the
          # first frame.
          sel_indices = tf.random_shuffle(
              tf.range(1, total_num_frames))[:(num_frames_per_video - 1)]
        # Concat first frame as reference frame to the front.
        sel_indices = tf.concat([tf.constant(0, dtype=tf.int32)[tf.newaxis],
                                 sel_indices], axis=0)
    else:
      if sample_adjacent_and_consistent_query_frames:
        if add_prev_frame_label:
          # Sample one more frame which we can use to provide initial softmax
          # feedback.
          num_frames_per_video += 1
        ref_idx = tf.random_shuffle(tf.range(total_num_frames))[0]
        sampling_is_valid = tf.greater_equal(total_num_frames,
                                             num_frames_per_video)
        def sample_query_start_idx():
          return tf.random_shuffle(
              tf.range(total_num_frames - num_frames_per_video + 1))[0]
        query_start_idx = tf.cond(sampling_is_valid, sample_query_start_idx,
                                  lambda: tf.constant(0, dtype=tf.int32))
        def sample_sel_indices():
          return tf.concat(
              [ref_idx[tf.newaxis],
               tf.range(
                   query_start_idx,
                   query_start_idx + (num_frames_per_video - 1))], axis=0)
        sel_indices = tf.cond(
            sampling_is_valid, sample_sel_indices,
            lambda: tf.zeros((num_frames_per_video,), dtype=tf.int32))
      else:
        # Randomly sample some frames from the video.
        sel_indices = tf.random_shuffle(
            tf.range(total_num_frames))[:num_frames_per_video]
    image = tf.gather(image, sel_indices, axis=0)
  if not video_frames_are_decoded:
    image = decode_image_sequence(image)

  if label is not None:
    if num_frames_per_video is not None:
      label = tf.gather(label, sel_indices, axis=0)
    if not video_frames_are_decoded:
      label = decode_image_sequence(label, image_format='png', channels=1)

    # Sometimes, label is saved as [num_frames_per_video, height, width] or
    # [num_frames_per_video, height, width, 1]. We change it to be
    # [num_frames_per_video, height, width, 1].
    if label.shape.ndims == 3:
      label = tf.expand_dims(label, 3)
    elif label.shape.ndims == 4 and label.shape.dims[3] == 1:
      pass
    else:
      raise ValueError('Input label shape must be '
                       '[num_frames_per_video, height, width],'
                       ' or [num_frames, height, width, 1]. '
                       'Got {}'.format(label.shape.ndims))
    label.set_shape([None, None, None, 1])

  # Add size of first dimension since tf can't figure it out automatically.
  image.set_shape((num_frames_per_video, None, None, None))
  if label is not None:
    label.set_shape((num_frames_per_video, None, None, None))

  preceding_frame_label = None
  if preprocess_image_and_label:
    if num_frames_per_video is None:
      raise ValueError('num_frame_per_video must be specified for preproc.')
    original_images = []
    images = []
    labels = []
    if sample_adjacent_and_consistent_query_frames:
      num_frames_individual_preproc = 1
    else:
      num_frames_individual_preproc = num_frames_per_video
    for frame_idx in range(num_frames_individual_preproc):
      original_image_t, image_t, label_t = (
          input_preprocess.preprocess_image_and_label(
              image[frame_idx],
              label[frame_idx],
              crop_height=crop_size[0] if crop_size is not None else None,
              crop_width=crop_size[1] if crop_size is not None else None,
              min_resize_value=min_resize_value,
              max_resize_value=max_resize_value,
              resize_factor=resize_factor,
              min_scale_factor=min_scale_factor,
              max_scale_factor=max_scale_factor,
              scale_factor_step_size=scale_factor_step_size,
              ignore_label=dataset.ignore_label,
              is_training=is_training,
              model_variant=model_variant))
      original_images.append(original_image_t)
      images.append(image_t)
      labels.append(label_t)
    if sample_adjacent_and_consistent_query_frames:
      imgs_for_preproc = [image[frame_idx] for frame_idx in
                          range(1, num_frames_per_video)]
      labels_for_preproc = [label[frame_idx] for frame_idx in
                            range(1, num_frames_per_video)]
      original_image_rest, image_rest, label_rest = (
          input_preprocess.preprocess_images_and_labels_consistently(
              imgs_for_preproc,
              labels_for_preproc,
              crop_height=crop_size[0] if crop_size is not None else None,
              crop_width=crop_size[1] if crop_size is not None else None,
              min_resize_value=min_resize_value,
              max_resize_value=max_resize_value,
              resize_factor=resize_factor,
              min_scale_factor=min_scale_factor,
              max_scale_factor=max_scale_factor,
              scale_factor_step_size=scale_factor_step_size,
              ignore_label=dataset.ignore_label,
              is_training=is_training,
              model_variant=model_variant))
      original_images.extend(original_image_rest)
      images.extend(image_rest)
      labels.extend(label_rest)
    assert len(original_images) == num_frames_per_video
    assert len(images) == num_frames_per_video
    assert len(labels) == num_frames_per_video

    if remap_labels_to_reference_frame:
      # Remap labels to indices into the labels of the (downscaled) reference
      # frame, or 0, i.e. background, for labels which are not present
      # in the reference.
      reference_labels = labels[0][tf.newaxis]
      h, w = train_utils.resolve_shape(reference_labels)[1:3]
      embedding_height = model.scale_dimension(
          h, 1.0 / decoder_output_stride)
      embedding_width = model.scale_dimension(
          w, 1.0 / decoder_output_stride)
      reference_labels_embedding_size = tf.squeeze(
          tf.image.resize_nearest_neighbor(
              reference_labels, tf.stack([embedding_height, embedding_width]),
              align_corners=True),
          axis=0)
      # Get sorted unique labels in the reference frame.
      labels_in_ref_frame, _ = tf.unique(
          tf.reshape(reference_labels_embedding_size, [-1]))
      labels_in_ref_frame = tf.contrib.framework.sort(labels_in_ref_frame)
      for idx in range(1, len(labels)):
        ref_label_mask = tf.equal(
            labels[idx],
            labels_in_ref_frame[tf.newaxis, tf.newaxis, :])
        remapped = tf.argmax(tf.cast(ref_label_mask, tf.uint8), axis=-1,
                             output_type=tf.int32)
        # Set to 0 if label is not present
        is_in_ref = tf.reduce_any(ref_label_mask, axis=-1)
        remapped *= tf.cast(is_in_ref, tf.int32)
        labels[idx] = remapped[..., tf.newaxis]

    if sample_adjacent_and_consistent_query_frames:
      if first_frame_finetuning and generate_prev_frame_mask_by_mask_damaging:
        preceding_frame_label = mask_damaging.damage_masks(labels[1])
      elif add_prev_frame_label:
        # Discard the image of the additional frame and take the label as
        # initialization for softmax feedback.
        original_images = [original_images[0]] + original_images[2:]
        preceding_frame_label = labels[1]
        images = [images[0]] + images[2:]
        labels = [labels[0]] + labels[2:]
        num_frames_per_video -= 1

    original_image = tf.stack(original_images, axis=0)
    image = tf.stack(images, axis=0)
    label = tf.stack(labels, axis=0)
  else:
    if label is not None:
      # Need to set label shape due to batching.
      label.set_shape([num_frames_per_video,
                       None if crop_size is None else crop_size[0],
                       None if crop_size is None else crop_size[1],
                       1])
    original_image = tf.to_float(tf.zeros_like(label))
    if crop_size is None:
      height = tf.shape(image)[1]
      width = tf.shape(image)[2]
    else:
      height = crop_size[0]
      width = crop_size[1]

  sample = {'image': image,
            'image_name': image_name,
            'height': height,
            'width': width,
            'video_id': video_id}
  if label is not None:
    sample['label'] = label

  if object_label is not None:
    sample['object_label'] = object_label

  if preceding_frame_label is not None:
    sample['preceding_frame_label'] = preceding_frame_label

  if not is_training:
    # Original image is only used during visualization.
    sample['original_image'] = original_image

  if is_training:
    if first_frame_finetuning:
      keep_input = tf.constant(True)
    else:
      keep_input = tf.logical_and(sampling_is_valid, tf.logical_and(
          _has_enough_pixels_of_each_object_in_first_frame(
              label, decoder_output_stride),
          _has_foreground_and_background_in_first_frame_2(
              label, decoder_output_stride)))

    batched = tf.train.maybe_batch(sample,
                                   keep_input=keep_input,
                                   batch_size=batch_size,
                                   num_threads=num_threads,
                                   capacity=batch_capacity_factor * batch_size,
                                   dynamic_pad=True)
  else:
    batched = tf.train.batch(sample,
                             batch_size=batch_size,
                             num_threads=num_threads,
                             capacity=batch_capacity_factor * batch_size,
                             dynamic_pad=True)

  # Flatten from [batch, num_frames_per_video, ...] to
  # batch * num_frames_per_video, ...].
  cropped_height = train_utils.resolve_shape(batched['image'])[2]
  cropped_width = train_utils.resolve_shape(batched['image'])[3]
  if num_frames_per_video is None:
    first_dim = -1
  else:
    first_dim = batch_size * num_frames_per_video
  batched['image'] = tf.reshape(batched['image'],
                                [first_dim, cropped_height, cropped_width, 3])
  if label is not None:
    batched['label'] = tf.reshape(batched['label'],
                                  [first_dim, cropped_height, cropped_width, 1])
  return batched
