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
r"""A Beam job to add contextual memory banks to tf.Examples.

This tool groups images containing bounding boxes and embedded context features
by a key, either `image/location` or `image/seq_id`, and time horizon,
then uses these groups to build up a contextual memory bank from the embedded
context features from each image in the group and adds that context to the
output tf.Examples for each image in the group.

Steps to generate a dataset with context from one with bounding boxes and
embedded context features:
1. Use object/detection/export_inference_graph.py to get a `saved_model` for
  inference. The input node must accept a tf.Example proto.
2. Run this tool with `saved_model` from step 1 and a TFRecord of tf.Example
  protos containing images, bounding boxes, and embedded context features.
  The context features can be added to tf.Examples using
  generate_embedding_data.py.

Example Usage:
--------------
python add_context_to_examples.py \
  --input_tfrecord path/to/input_tfrecords* \
  --output_tfrecord path/to/output_tfrecords \
  --sequence_key image/location \
  --time_horizon month

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import datetime
import io
import itertools
import json
import os
import numpy as np
import PIL.Image
import six
import tensorflow as tf

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass


class ReKeyDataFn(beam.DoFn):
  """Re-keys tfrecords by sequence_key.

  This Beam DoFn re-keys the tfrecords by a user-defined sequence_key
  """

  def __init__(self, sequence_key, time_horizon,
               reduce_image_size, max_image_dimension):
    """Initialization function.

    Args:
      sequence_key: A feature name to use as a key for grouping sequences.
        Must point to a key of type bytes_list
      time_horizon: What length of time to use to partition the data when
        building the memory banks. Options: `year`, `month`, `week`, `day `,
        `hour`, `minute`, None
      reduce_image_size: Whether to reduce the sizes of the stored images.
      max_image_dimension: maximum dimension of reduced images
    """
    self._sequence_key = sequence_key
    if time_horizon is None or time_horizon in {'year', 'month', 'week', 'day',
                                                'hour', 'minute'}:
      self._time_horizon = time_horizon
    else:
      raise ValueError('Time horizon not supported.')
    self._reduce_image_size = reduce_image_size
    self._max_image_dimension = max_image_dimension
    self._session = None
    self._num_examples_processed = beam.metrics.Metrics.counter(
        'data_rekey', 'num_tf_examples_processed')
    self._num_images_resized = beam.metrics.Metrics.counter(
        'data_rekey', 'num_images_resized')
    self._num_images_read = beam.metrics.Metrics.counter(
        'data_rekey', 'num_images_read')
    self._num_images_found = beam.metrics.Metrics.counter(
        'data_rekey', 'num_images_read')
    self._num_got_shape = beam.metrics.Metrics.counter(
        'data_rekey', 'num_images_got_shape')
    self._num_images_found_size = beam.metrics.Metrics.counter(
        'data_rekey', 'num_images_found_size')
    self._num_examples_cleared = beam.metrics.Metrics.counter(
        'data_rekey', 'num_examples_cleared')
    self._num_examples_updated = beam.metrics.Metrics.counter(
        'data_rekey', 'num_examples_updated')

  def process(self, tfrecord_entry):
    return self._rekey_examples(tfrecord_entry)

  def _largest_size_at_most(self, height, width, largest_side):
    """Computes new shape with the largest side equal to `largest_side`.

    Args:
      height: an int indicating the current height.
      width: an int indicating the current width.
      largest_side: A python integer indicating the size of
        the largest side after resize.
    Returns:
      new_height: an int indicating the new height.
      new_width: an int indicating the new width.
    """

    x_scale = float(largest_side) / float(width)
    y_scale = float(largest_side) / float(height)
    scale = min(x_scale, y_scale)

    new_width = int(width * scale)
    new_height = int(height * scale)

    return new_height, new_width

  def _resize_image(self, input_example):
    """Resizes the image within input_example and updates the height and width.

    Args:
      input_example: A tf.Example that we want to update to contain a resized
        image.
    Returns:
      input_example: Updated tf.Example.
    """

    original_image = copy.deepcopy(
        input_example.features.feature['image/encoded'].bytes_list.value[0])
    self._num_images_read.inc(1)

    height = copy.deepcopy(
        input_example.features.feature['image/height'].int64_list.value[0])

    width = copy.deepcopy(
        input_example.features.feature['image/width'].int64_list.value[0])

    self._num_got_shape.inc(1)

    new_height, new_width = self._largest_size_at_most(
        height, width, self._max_image_dimension)

    self._num_images_found_size.inc(1)

    encoded_jpg_io = io.BytesIO(original_image)
    image = PIL.Image.open(encoded_jpg_io)
    resized_image = image.resize((new_width, new_height))

    with io.BytesIO() as output:
      resized_image.save(output, format='JPEG')
      encoded_resized_image = output.getvalue()

    self._num_images_resized.inc(1)

    del input_example.features.feature['image/encoded'].bytes_list.value[:]
    del input_example.features.feature['image/height'].int64_list.value[:]
    del input_example.features.feature['image/width'].int64_list.value[:]

    self._num_examples_cleared.inc(1)

    input_example.features.feature['image/encoded'].bytes_list.value.extend(
        [encoded_resized_image])
    input_example.features.feature['image/height'].int64_list.value.extend(
        [new_height])
    input_example.features.feature['image/width'].int64_list.value.extend(
        [new_width])
    self._num_examples_updated.inc(1)

    return input_example

  def _rekey_examples(self, tfrecord_entry):
    serialized_example = copy.deepcopy(tfrecord_entry)

    input_example = tf.train.Example.FromString(serialized_example)

    self._num_images_found.inc(1)

    if self._reduce_image_size:
      input_example = self._resize_image(input_example)
      self._num_images_resized.inc(1)

    new_key = input_example.features.feature[
        self._sequence_key].bytes_list.value[0]

    if self._time_horizon:
      date_captured = datetime.datetime.strptime(
          six.ensure_str(input_example.features.feature[
              'image/date_captured'].bytes_list.value[0]), '%Y-%m-%d %H:%M:%S')
      year = date_captured.year
      month = date_captured.month
      day = date_captured.day
      week = np.floor(float(day) / float(7))
      hour = date_captured.hour
      minute = date_captured.minute

      if self._time_horizon == 'year':
        new_key = new_key + six.ensure_binary('/' + str(year))
      elif self._time_horizon == 'month':
        new_key = new_key + six.ensure_binary(
            '/' + str(year) + '/' + str(month))
      elif self._time_horizon == 'week':
        new_key = new_key + six.ensure_binary(
            '/' + str(year) + '/' + str(month) + '/' + str(week))
      elif self._time_horizon == 'day':
        new_key = new_key + six.ensure_binary(
            '/' + str(year) + '/' + str(month) + '/' + str(day))
      elif self._time_horizon == 'hour':
        new_key = new_key + six.ensure_binary(
            '/' + str(year) + '/' + str(month) + '/' + str(day) + '/' + (
                str(hour)))
      elif self._time_horizon == 'minute':
        new_key = new_key + six.ensure_binary(
            '/' + str(year) + '/' + str(month) + '/' + str(day) + '/' + (
                str(hour) + '/' + str(minute)))

    self._num_examples_processed.inc(1)

    return [(new_key, input_example)]


class SortGroupedDataFn(beam.DoFn):
  """Sorts data within a keyed group.

  This Beam DoFn sorts the grouped list of image examples by frame_num
  """

  def __init__(self, sequence_key, sorted_image_ids,
               max_num_elements_in_context_features):
    """Initialization function.

    Args:
      sequence_key: A feature name to use as a key for grouping sequences.
        Must point to a key of type bytes_list
      sorted_image_ids: Whether the image ids are sortable to use as sorting
        tie-breakers
      max_num_elements_in_context_features: The maximum number of elements
        allowed in the memory bank
    """
    self._session = None
    self._num_examples_processed = beam.metrics.Metrics.counter(
        'sort_group', 'num_groups_sorted')
    self._too_many_elements = beam.metrics.Metrics.counter(
        'sort_group', 'too_many_elements')
    self._split_elements = beam.metrics.Metrics.counter(
        'sort_group', 'split_elements')
    self._sequence_key = six.ensure_binary(sequence_key)
    self._sorted_image_ids = sorted_image_ids
    self._max_num_elements_in_context_features = (
        max_num_elements_in_context_features)

  def process(self, grouped_entry):
    return self._sort_image_examples(grouped_entry)

  def _sort_image_examples(self, grouped_entry):
    key, example_collection = grouped_entry
    example_list = list(example_collection)

    def get_frame_num(example):
      return example.features.feature['image/seq_frame_num'].int64_list.value[0]

    def get_date_captured(example):
      return datetime.datetime.strptime(
          six.ensure_str(
              example.features.feature[
                  'image/date_captured'].bytes_list.value[0]),
          '%Y-%m-%d %H:%M:%S')

    def get_image_id(example):
      return example.features.feature['image/source_id'].bytes_list.value[0]

    if self._sequence_key == six.ensure_binary('image/seq_id'):
      sorting_fn = get_frame_num
    elif self._sequence_key == six.ensure_binary('image/location'):
      if self._sorted_image_ids:
        sorting_fn = get_image_id
      else:
        sorting_fn = get_date_captured

    sorted_example_list = sorted(example_list, key=sorting_fn)

    self._num_examples_processed.inc(1)

    if len(sorted_example_list) > self._max_num_elements_in_context_features:
      leftovers = sorted_example_list
      output_list = []
      count = 0
      self._too_many_elements.inc(1)
      while len(leftovers) > self._max_num_elements_in_context_features:
        self._split_elements.inc(1)
        new_key = key + six.ensure_binary('_' + str(count))
        new_list = leftovers[:self._max_num_elements_in_context_features]
        output_list.append((new_key, new_list))
        leftovers = leftovers[:self._max_num_elements_in_context_features]
        count += 1
    else:
      output_list = [(key, sorted_example_list)]

    return output_list


def get_sliding_window(example_list, max_clip_length, stride_length):
  """Yields a sliding window over data from example_list.

  Sliding window has width max_clip_len (n) and stride stride_len (m).
     s -> (s0,s1,...s[n-1]), (s[m],s[m+1],...,s[m+n]), ...

  Args:
    example_list: A list of examples.
    max_clip_length: The maximum length of each clip.
    stride_length: The stride between each clip.

  Yields:
    A list of lists of examples, each with length <= max_clip_length
  """

  # check if the list is too short to slide over
  if len(example_list) < max_clip_length:
    yield example_list
  else:
    starting_values = [i*stride_length for i in
                       range(len(example_list)) if
                       len(example_list) > i*stride_length]
    for start in starting_values:
      result = tuple(itertools.islice(example_list, start,
                                      min(start + max_clip_length,
                                          len(example_list))))
      yield result


class GenerateContextFn(beam.DoFn):
  """Generates context data for camera trap images.

  This Beam DoFn builds up contextual memory banks from groups of images and
  stores them in the output tf.Example or tf.Sequence_example for each image.
  """

  def __init__(self, sequence_key, add_context_features, image_ids_to_keep,
               keep_context_features_image_id_list=False,
               subsample_context_features_rate=0,
               keep_only_positives=False,
               context_features_score_threshold=0.7,
               keep_only_positives_gt=False,
               max_num_elements_in_context_features=5000,
               pad_context_features=False,
               output_type='tf_example', max_clip_length=None,
               context_feature_length=2057):
    """Initialization function.

    Args:
      sequence_key: A feature name to use as a key for grouping sequences.
      add_context_features: Whether to keep and store the contextual memory
        bank.
      image_ids_to_keep: A list of image ids to save, to use to build data
        subsets for evaluation.
      keep_context_features_image_id_list: Whether to save an ordered list of
        the ids of the images in the contextual memory bank.
      subsample_context_features_rate: What rate to subsample images for the
        contextual memory bank.
      keep_only_positives: Whether to only keep high scoring
        (>context_features_score_threshold) features in the contextual memory
        bank.
      context_features_score_threshold: What threshold to use for keeping
        features.
      keep_only_positives_gt: Whether to only keep features from images that
        contain objects based on the ground truth (for training).
      max_num_elements_in_context_features: the maximum number of elements in
        the memory bank
      pad_context_features: Whether to pad the context features to a fixed size.
      output_type: What type of output, tf_example of tf_sequence_example
      max_clip_length: The maximum length of a sequence example, before
        splitting into multiple
      context_feature_length: The length of the context feature embeddings
        stored in the input data.
    """
    self._session = None
    self._num_examples_processed = beam.metrics.Metrics.counter(
        'sequence_data_generation', 'num_seq_examples_processed')
    self._num_keys_processed = beam.metrics.Metrics.counter(
        'sequence_data_generation', 'num_keys_processed')
    self._sequence_key = sequence_key
    self._add_context_features = add_context_features
    self._pad_context_features = pad_context_features
    self._output_type = output_type
    self._max_clip_length = max_clip_length
    if six.ensure_str(image_ids_to_keep) == 'All':
      self._image_ids_to_keep = None
    else:
      with tf.io.gfile.GFile(image_ids_to_keep) as f:
        self._image_ids_to_keep = json.load(f)
    self._keep_context_features_image_id_list = (
        keep_context_features_image_id_list)
    self._subsample_context_features_rate = subsample_context_features_rate
    self._keep_only_positives = keep_only_positives
    self._keep_only_positives_gt = keep_only_positives_gt
    self._context_features_score_threshold = context_features_score_threshold
    self._max_num_elements_in_context_features = (
        max_num_elements_in_context_features)
    self._context_feature_length = context_feature_length

    self._images_kept = beam.metrics.Metrics.counter(
        'sequence_data_generation', 'images_kept')
    self._images_loaded = beam.metrics.Metrics.counter(
        'sequence_data_generation', 'images_loaded')

  def process(self, grouped_entry):
    return self._add_context_to_example(copy.deepcopy(grouped_entry))

  def _build_context_features(self, example_list):
    context_features = []
    context_features_image_id_list = []
    count = 0
    example_embedding = []

    for idx, example in enumerate(example_list):
      if self._subsample_context_features_rate > 0:
        if (idx % self._subsample_context_features_rate) != 0:
          example.features.feature[
              'context_features_idx'].int64_list.value.append(
                  self._max_num_elements_in_context_features + 1)
          continue
      if self._keep_only_positives:
        if example.features.feature[
            'image/embedding_score'
            ].float_list.value[0] < self._context_features_score_threshold:
          example.features.feature[
              'context_features_idx'].int64_list.value.append(
                  self._max_num_elements_in_context_features + 1)
          continue
      if self._keep_only_positives_gt:
        if len(example.features.feature[
            'image/object/bbox/xmin'
            ].float_list.value) < 1:
          example.features.feature[
              'context_features_idx'].int64_list.value.append(
                  self._max_num_elements_in_context_features + 1)
          continue

      example_embedding = list(example.features.feature[
          'image/embedding'].float_list.value)
      context_features.extend(example_embedding)
      example.features.feature[
          'context_features_idx'].int64_list.value.append(count)
      count += 1
      example_image_id = example.features.feature[
          'image/source_id'].bytes_list.value[0]
      context_features_image_id_list.append(example_image_id)

    if not example_embedding:
      example_embedding.append(np.zeros(self._context_feature_length))

    feature_length = self._context_feature_length

    # If the example_list is not empty and image/embedding_length is in the
    # featture dict, feature_length will be assigned to that. Otherwise, it will
    # be kept as default.
    if example_list and (
        'image/embedding_length' in example_list[0].features.feature):
      feature_length = example_list[0].features.feature[
          'image/embedding_length'].int64_list.value[0]

    if self._pad_context_features:
      while len(context_features_image_id_list) < (
          self._max_num_elements_in_context_features):
        context_features_image_id_list.append('')

    return context_features, feature_length, context_features_image_id_list

  def _add_context_to_example(self, grouped_entry):
    key, example_collection = grouped_entry
    list_of_examples = []

    example_list = list(example_collection)

    if self._add_context_features:
      context_features, feature_length, context_features_image_id_list = (
          self._build_context_features(example_list))

    if self._image_ids_to_keep is not None:
      new_example_list = []
      for example in example_list:
        im_id = example.features.feature['image/source_id'].bytes_list.value[0]
        self._images_loaded.inc(1)
        if six.ensure_str(im_id) in self._image_ids_to_keep:
          self._images_kept.inc(1)
          new_example_list.append(example)
      if new_example_list:
        example_list = new_example_list
      else:
        return []

    if self._output_type == 'tf_sequence_example':
      if self._max_clip_length is not None:
        # For now, no overlap
        clips = get_sliding_window(
            example_list, self._max_clip_length, self._max_clip_length)
      else:
        clips = [example_list]

      for clip_num, clip_list in enumerate(clips):
        # initialize sequence example
        seq_example = tf.train.SequenceExample()
        video_id = six.ensure_str(key)+'_'+ str(clip_num)
        seq_example.context.feature['clip/media_id'].bytes_list.value.append(
            video_id.encode('utf8'))
        seq_example.context.feature['clip/frames'].int64_list.value.append(
            len(clip_list))

        seq_example.context.feature[
            'clip/start/timestamp'].int64_list.value.append(0)
        seq_example.context.feature[
            'clip/end/timestamp'].int64_list.value.append(len(clip_list))
        seq_example.context.feature['image/format'].bytes_list.value.append(
            six.ensure_binary('JPG'))
        seq_example.context.feature['image/channels'].int64_list.value.append(3)
        context_example = clip_list[0]
        seq_example.context.feature['image/height'].int64_list.value.append(
            context_example.features.feature[
                'image/height'].int64_list.value[0])
        seq_example.context.feature['image/width'].int64_list.value.append(
            context_example.features.feature['image/width'].int64_list.value[0])

        seq_example.context.feature[
            'image/context_feature_length'].int64_list.value.append(
                feature_length)
        seq_example.context.feature[
            'image/context_features'].float_list.value.extend(
                context_features)
        if self._keep_context_features_image_id_list:
          seq_example.context.feature[
              'image/context_features_image_id_list'].bytes_list.value.extend(
                  context_features_image_id_list)

        encoded_image_list = seq_example.feature_lists.feature_list[
            'image/encoded']
        timestamps_list = seq_example.feature_lists.feature_list[
            'image/timestamp']
        context_features_idx_list = seq_example.feature_lists.feature_list[
            'image/context_features_idx']
        date_captured_list = seq_example.feature_lists.feature_list[
            'image/date_captured']
        unix_time_list = seq_example.feature_lists.feature_list[
            'image/unix_time']
        location_list = seq_example.feature_lists.feature_list['image/location']
        image_ids_list = seq_example.feature_lists.feature_list[
            'image/source_id']
        gt_xmin_list = seq_example.feature_lists.feature_list[
            'region/bbox/xmin']
        gt_xmax_list = seq_example.feature_lists.feature_list[
            'region/bbox/xmax']
        gt_ymin_list = seq_example.feature_lists.feature_list[
            'region/bbox/ymin']
        gt_ymax_list = seq_example.feature_lists.feature_list[
            'region/bbox/ymax']
        gt_type_list = seq_example.feature_lists.feature_list[
            'region/label/index']
        gt_type_string_list = seq_example.feature_lists.feature_list[
            'region/label/string']
        gt_is_annotated_list = seq_example.feature_lists.feature_list[
            'region/is_annotated']

        for idx, example in enumerate(clip_list):

          encoded_image = encoded_image_list.feature.add()
          encoded_image.bytes_list.value.extend(
              example.features.feature['image/encoded'].bytes_list.value)

          image_id = image_ids_list.feature.add()
          image_id.bytes_list.value.append(
              example.features.feature['image/source_id'].bytes_list.value[0])

          timestamp = timestamps_list.feature.add()
          # Timestamp is currently order in the list.
          timestamp.int64_list.value.extend([idx])

          context_features_idx = context_features_idx_list.feature.add()
          context_features_idx.int64_list.value.extend(
              example.features.feature['context_features_idx'].int64_list.value)

          date_captured = date_captured_list.feature.add()
          date_captured.bytes_list.value.extend(
              example.features.feature['image/date_captured'].bytes_list.value)
          unix_time = unix_time_list.feature.add()
          unix_time.float_list.value.extend(
              example.features.feature['image/unix_time'].float_list.value)
          location = location_list.feature.add()
          location.bytes_list.value.extend(
              example.features.feature['image/location'].bytes_list.value)

          gt_xmin = gt_xmin_list.feature.add()
          gt_xmax = gt_xmax_list.feature.add()
          gt_ymin = gt_ymin_list.feature.add()
          gt_ymax = gt_ymax_list.feature.add()
          gt_type = gt_type_list.feature.add()
          gt_type_str = gt_type_string_list.feature.add()

          gt_is_annotated = gt_is_annotated_list.feature.add()
          gt_is_annotated.int64_list.value.append(1)

          gt_xmin.float_list.value.extend(
              example.features.feature[
                  'image/object/bbox/xmin'].float_list.value)
          gt_xmax.float_list.value.extend(
              example.features.feature[
                  'image/object/bbox/xmax'].float_list.value)
          gt_ymin.float_list.value.extend(
              example.features.feature[
                  'image/object/bbox/ymin'].float_list.value)
          gt_ymax.float_list.value.extend(
              example.features.feature[
                  'image/object/bbox/ymax'].float_list.value)

          gt_type.int64_list.value.extend(
              example.features.feature[
                  'image/object/class/label'].int64_list.value)
          gt_type_str.bytes_list.value.extend(
              example.features.feature[
                  'image/object/class/text'].bytes_list.value)

        self._num_examples_processed.inc(1)
        list_of_examples.append(seq_example)

    elif self._output_type == 'tf_example':

      for example in example_list:
        im_id = example.features.feature['image/source_id'].bytes_list.value[0]

        if self._add_context_features:
          example.features.feature[
              'image/context_features'].float_list.value.extend(
                  context_features)
          example.features.feature[
              'image/context_feature_length'].int64_list.value.append(
                  feature_length)

        if self._keep_context_features_image_id_list:
          example.features.feature[
              'image/context_features_image_id_list'].bytes_list.value.extend(
                  context_features_image_id_list)

        self._num_examples_processed.inc(1)
        list_of_examples.append(example)

    return list_of_examples


def construct_pipeline(pipeline,
                       input_tfrecord,
                       output_tfrecord,
                       sequence_key,
                       time_horizon=None,
                       subsample_context_features_rate=0,
                       reduce_image_size=True,
                       max_image_dimension=1024,
                       add_context_features=True,
                       sorted_image_ids=True,
                       image_ids_to_keep='All',
                       keep_context_features_image_id_list=False,
                       keep_only_positives=False,
                       context_features_score_threshold=0.7,
                       keep_only_positives_gt=False,
                       max_num_elements_in_context_features=5000,
                       num_shards=0,
                       output_type='tf_example',
                       max_clip_length=None,
                       context_feature_length=2057):
  """Returns a beam pipeline to run object detection inference.

  Args:
    pipeline: Initialized beam pipeline.
    input_tfrecord: An TFRecord of tf.train.Example protos containing images.
    output_tfrecord: An TFRecord of tf.train.Example protos that contain images
      in the input TFRecord and the detections from the model.
    sequence_key: A feature name to use as a key for grouping sequences.
    time_horizon: What length of time to use to partition the data when building
      the memory banks. Options: `year`, `month`, `week`, `day `, `hour`,
      `minute`, None.
    subsample_context_features_rate: What rate to subsample images for the
        contextual memory bank.
    reduce_image_size: Whether to reduce the size of the stored images.
    max_image_dimension: The maximum image dimension to use for resizing.
    add_context_features: Whether to keep and store the contextual memory bank.
    sorted_image_ids: Whether the image ids are sortable, and can be used as
      datetime tie-breakers when building memory banks.
    image_ids_to_keep: A list of image ids to save, to use to build data subsets
      for evaluation.
    keep_context_features_image_id_list: Whether to save an ordered list of the
      ids of the images in the contextual memory bank.
    keep_only_positives: Whether to only keep high scoring
      (>context_features_score_threshold) features in the contextual memory
      bank.
    context_features_score_threshold: What threshold to use for keeping
      features.
    keep_only_positives_gt: Whether to only keep features from images that
      contain objects based on the ground truth (for training).
    max_num_elements_in_context_features: the maximum number of elements in the
      memory bank
    num_shards: The number of output shards.
    output_type: What type of output, tf_example of tf_sequence_example
    max_clip_length: The maximum length of a sequence example, before
      splitting into multiple
    context_feature_length: The length of the context feature embeddings stored
      in the input data.
  """
  if output_type == 'tf_example':
    coder = beam.coders.ProtoCoder(tf.train.Example)
  elif output_type == 'tf_sequence_example':
    coder = beam.coders.ProtoCoder(tf.train.SequenceExample)
  else:
    raise ValueError('Unsupported output type.')
  input_collection = (
      pipeline | 'ReadInputTFRecord' >> beam.io.tfrecordio.ReadFromTFRecord(
          input_tfrecord,
          coder=beam.coders.BytesCoder()))
  rekey_collection = input_collection | 'RekeyExamples' >> beam.ParDo(
      ReKeyDataFn(sequence_key, time_horizon,
                  reduce_image_size, max_image_dimension))
  grouped_collection = (
      rekey_collection | 'GroupBySequenceKey' >> beam.GroupByKey())
  grouped_collection = (
      grouped_collection | 'ReshuffleGroups' >> beam.Reshuffle())
  ordered_collection = (
      grouped_collection | 'OrderByFrameNumber' >> beam.ParDo(
          SortGroupedDataFn(sequence_key, sorted_image_ids,
                            max_num_elements_in_context_features)))
  ordered_collection = (
      ordered_collection | 'ReshuffleSortedGroups' >> beam.Reshuffle())
  output_collection = (
      ordered_collection | 'AddContextToExamples' >> beam.ParDo(
          GenerateContextFn(
              sequence_key, add_context_features, image_ids_to_keep,
              keep_context_features_image_id_list=(
                  keep_context_features_image_id_list),
              subsample_context_features_rate=subsample_context_features_rate,
              keep_only_positives=keep_only_positives,
              keep_only_positives_gt=keep_only_positives_gt,
              context_features_score_threshold=(
                  context_features_score_threshold),
              max_num_elements_in_context_features=(
                  max_num_elements_in_context_features),
              output_type=output_type,
              max_clip_length=max_clip_length,
              context_feature_length=context_feature_length)))

  output_collection = (
      output_collection | 'ReshuffleExamples' >> beam.Reshuffle())
  _ = output_collection | 'WritetoDisk' >> beam.io.tfrecordio.WriteToTFRecord(
      output_tfrecord,
      num_shards=num_shards,
      coder=coder)


def parse_args(argv):
  """Command-line argument parser.

  Args:
    argv: command line arguments
  Returns:
    beam_args: Arguments for the beam pipeline.
    pipeline_args: Arguments for the pipeline options, such as runner type.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_tfrecord',
      dest='input_tfrecord',
      required=True,
      help='TFRecord containing images in tf.Example format for object '
      'detection, with bounding boxes and contextual feature embeddings.')
  parser.add_argument(
      '--output_tfrecord',
      dest='output_tfrecord',
      required=True,
      help='TFRecord containing images in tf.Example format, with added '
      'contextual memory banks.')
  parser.add_argument(
      '--sequence_key',
      dest='sequence_key',
      default='image/location',
      help='Key to use when grouping sequences: so far supports `image/seq_id` '
      'and `image/location`.')
  parser.add_argument(
      '--context_feature_length',
      dest='context_feature_length',
      default=2057,
      help='The length of the context feature embeddings stored in the input '
      'data.')
  parser.add_argument(
      '--time_horizon',
      dest='time_horizon',
      default=None,
      help='What time horizon to use when splitting the data, if any. Options '
      'are: `year`, `month`, `week`, `day `, `hour`, `minute`, `None`.')
  parser.add_argument(
      '--subsample_context_features_rate',
      dest='subsample_context_features_rate',
      default=0,
      help='Whether to subsample the context_features, and if so how many to '
      'sample. If the rate is set to X, it will sample context from 1 out of '
      'every X images. Default is sampling from every image, which is X=0.')
  parser.add_argument(
      '--reduce_image_size',
      dest='reduce_image_size',
      default=True,
      help='downsamples images to have longest side max_image_dimension, '
      'maintaining aspect ratio')
  parser.add_argument(
      '--max_image_dimension',
      dest='max_image_dimension',
      default=1024,
      help='Sets max image dimension for resizing.')
  parser.add_argument(
      '--add_context_features',
      dest='add_context_features',
      default=True,
      help='Adds a memory bank of embeddings to each clip')
  parser.add_argument(
      '--sorted_image_ids',
      dest='sorted_image_ids',
      default=True,
      help='Whether the image source_ids are sortable to deal with '
      'date_captured tie-breaks.')
  parser.add_argument(
      '--image_ids_to_keep',
      dest='image_ids_to_keep',
      default='All',
      help='Path to .json list of image ids to keep, used for ground truth '
      'eval creation.')
  parser.add_argument(
      '--keep_context_features_image_id_list',
      dest='keep_context_features_image_id_list',
      default=False,
      help='Whether or not to keep a list of the image_ids corresponding to '
      'the memory bank.')
  parser.add_argument(
      '--keep_only_positives',
      dest='keep_only_positives',
      default=False,
      help='Whether or not to keep only positive boxes based on score.')
  parser.add_argument(
      '--context_features_score_threshold',
      dest='context_features_score_threshold',
      default=0.7,
      help='What score threshold to use for boxes in context_features, when '
      '`keep_only_positives` is set to `True`.')
  parser.add_argument(
      '--keep_only_positives_gt',
      dest='keep_only_positives_gt',
      default=False,
      help='Whether or not to keep only positive boxes based on gt class.')
  parser.add_argument(
      '--max_num_elements_in_context_features',
      dest='max_num_elements_in_context_features',
      default=2000,
      help='Sets max number of context feature elements per memory bank. '
      'If the number of images in the context group is greater than '
      '`max_num_elements_in_context_features`, the context group will be split.'
      )
  parser.add_argument(
      '--output_type',
      dest='output_type',
      default='tf_example',
      help='Output type, one of `tf_example`, `tf_sequence_example`.')
  parser.add_argument(
      '--max_clip_length',
      dest='max_clip_length',
      default=None,
      help='Max length for sequence example outputs.')
  parser.add_argument(
      '--num_shards',
      dest='num_shards',
      default=0,
      help='Number of output shards.')
  beam_args, pipeline_args = parser.parse_known_args(argv)
  return beam_args, pipeline_args


def main(argv=None, save_main_session=True):
  """Runs the Beam pipeline that performs inference.

  Args:
    argv: Command line arguments.
    save_main_session: Whether to save the main session.
  """
  args, pipeline_args = parse_args(argv)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
            pipeline_args)
  pipeline_options.view_as(
      beam.options.pipeline_options.SetupOptions).save_main_session = (
          save_main_session)

  dirname = os.path.dirname(args.output_tfrecord)
  tf.io.gfile.makedirs(dirname)

  p = beam.Pipeline(options=pipeline_options)

  construct_pipeline(
      p,
      args.input_tfrecord,
      args.output_tfrecord,
      args.sequence_key,
      args.time_horizon,
      args.subsample_context_features_rate,
      args.reduce_image_size,
      args.max_image_dimension,
      args.add_context_features,
      args.sorted_image_ids,
      args.image_ids_to_keep,
      args.keep_context_features_image_id_list,
      args.keep_only_positives,
      args.context_features_score_threshold,
      args.keep_only_positives_gt,
      args.max_num_elements_in_context_features,
      args.num_shards,
      args.output_type,
      args.max_clip_length,
      args.context_feature_length)

  p.run()


if __name__ == '__main__':
  main()
