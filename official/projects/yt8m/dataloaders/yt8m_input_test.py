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

import os

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from official.core import input_reader
from official.projects.yt8m.configs import yt8m as yt8m_configs
from official.projects.yt8m.dataloaders import utils
from official.projects.yt8m.dataloaders import yt8m_input
from official.vision.dataloaders import tfexample_utils


class Yt8mInputTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._model_dir = os.path.join(self.get_temp_dir(), 'model_dir')
    tf.io.gfile.makedirs(self._model_dir)

    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self.data_path = os.path.join(data_dir, 'data.tfrecord')
    self.num_segment = 6
    examples = [
        utils.make_yt8m_example(self.num_segment, 120 + i) for i in range(8)
    ]
    tfexample_utils.dump_to_tfrecord(self.data_path, tf_examples=examples)

  def create_input_reader(self, params):
    decoder = yt8m_input.Decoder(input_params=params)
    decoder_fn = decoder.decode
    parser = yt8m_input.Parser(input_params=params)
    parser_fn = parser.parse_fn(params.is_training)
    postprocess = yt8m_input.PostBatchProcessor(input_params=params)
    postprocess_fn = postprocess.post_fn
    transform_batch = yt8m_input.TransformBatcher(input_params=params)
    batch_fn = transform_batch.batch_fn

    return input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder_fn,
        parser_fn=parser_fn,
        postprocess_fn=postprocess_fn,
        transform_and_batch_fn=batch_fn)

  @parameterized.parameters((True, 20), (False, 20), (False, None))
  def test_read_video_level_input(self, include_video_id, num_sample_frames):
    params = yt8m_configs.yt8m(is_training=False)
    params.global_batch_size = 4
    params.segment_labels = False
    params.input_path = self.data_path
    params.include_video_id = include_video_id
    params.max_frames = 122
    params.num_sample_frames = num_sample_frames
    reader = self.create_input_reader(params)

    dataset = reader.read()
    iterator = iter(dataset)
    example = next(iterator)

    for k, v in example.items():
      logging.info('DEBUG read example %r %r %r', k, v.shape, type(v))
    if include_video_id:
      self.assertCountEqual(
          ['video_matrix', 'labels', 'num_frames', 'video_ids'], example.keys())
    else:
      self.assertCountEqual(['video_matrix', 'labels', 'num_frames'],
                            example.keys())
    batch_size = params.global_batch_size
    expected_num_frames = num_sample_frames or params.max_frames
    self.assertEqual(
        example['video_matrix'].shape.as_list(),
        [batch_size, expected_num_frames, sum(params.feature_sizes)],
    )
    self.assertEqual(
        example['labels'].shape.as_list(), [batch_size, params.num_classes]
    )
    # Check non empty labels.
    self.assertGreater(np.nonzero(example['labels'][0].numpy())[0].shape[0], 0)

    if num_sample_frames:
      self.assertAllEqual(
          example['num_frames'].numpy(),
          [[num_sample_frames]] * batch_size,
      )
    else:
      self.assertAllEqual(
          example['num_frames'].numpy(),
          [[120], [121], [122], [122]],
      )

    if include_video_id:
      self.assertEqual(example['video_ids'].shape.as_list(), [batch_size, 1])

  @parameterized.parameters((True, 20), (False, 20), (False, None))
  def test_read_segment_level_input(self, include_video_id, num_sample_frames):
    params = yt8m_configs.yt8m(is_training=False)
    params.global_batch_size = 2
    params.segment_labels = True
    params.segment_size = 24
    params.input_path = self.data_path
    params.include_video_id = include_video_id
    params.num_sample_frames = num_sample_frames
    reader = self.create_input_reader(params)

    dataset = reader.read()
    iterator = iter(dataset)
    example = next(iterator)

    for k, v in example.items():
      logging.info('DEBUG read example %r %r %r', k, v.shape, type(v))
    if include_video_id:
      self.assertCountEqual([
          'video_matrix', 'labels', 'num_frames', 'label_weights', 'video_ids'
      ], example.keys())
    else:
      self.assertCountEqual(
          ['video_matrix', 'labels', 'num_frames', 'label_weights'],
          example.keys())
    batch_size = params.global_batch_size * self.num_segment
    expected_num_frames = num_sample_frames or params.max_frames
    self.assertEqual(
        example['video_matrix'].shape.as_list(),
        [batch_size, expected_num_frames, sum(params.feature_sizes)],
    )
    self.assertEqual(example['labels'].shape.as_list(),
                     [batch_size, params.num_classes])
    self.assertGreater(np.nonzero(example['labels'][0].numpy())[0].shape[0], 0)
    self.assertEqual(example['label_weights'].shape.as_list(),
                     [batch_size, params.num_classes])

    if num_sample_frames:
      self.assertAllEqual(
          example['num_frames'].numpy(),
          [[num_sample_frames]] * batch_size,
      )
    else:
      self.assertAllEqual(
          example['num_frames'].numpy(),
          [[params.segment_size]] * batch_size,
      )

    if include_video_id:
      self.assertEqual(example['video_ids'].shape.as_list(), [batch_size])

  @parameterized.parameters(
      (True, 4, False),
      (False, 4, False),
      (False, None, False),
      (True, 4, True),
      (False, 4, True),
      (False, None, True),
  )
  def test_read_video_level_float_input(
      self, include_video_id, num_sample_frames, per_feature_l2_norm
  ):
    data_dir = os.path.join(self.get_temp_dir(), 'data2')
    tf.io.gfile.makedirs(data_dir)
    data_path = os.path.join(data_dir, 'data2.tfrecord')
    examples = [
        utils.make_example_with_float_features(self.num_segment)
        for _ in range(8)
    ]
    tfexample_utils.dump_to_tfrecord(data_path, tf_examples=examples)

    params = yt8m_configs.yt8m(is_training=False)
    params.global_batch_size = 4
    params.segment_labels = False
    params.input_path = data_path
    params.num_frames = 2
    params.max_frames = 2
    params.num_sample_frames = num_sample_frames
    params.feature_names = ('VIDEO_EMBEDDING/context_feature/floats',
                            'FEATURE/feature/floats')
    params.feature_sources = ('context', 'feature')
    params.feature_dtypes = ('float32', 'float32')
    params.feature_sizes = (256, 2048)
    params.feature_from_bytes = (False, False)
    params.label_field = 'clip/label/index'
    params.include_video_id = include_video_id
    params.input_per_feature_l2_norm = per_feature_l2_norm
    reader = self.create_input_reader(params)

    dataset = reader.read()
    iterator = iter(dataset)
    example = next(iterator)

    for k, v in example.items():
      logging.info('DEBUG read example %r %r %r', k, v.shape, type(v))
    logging.info('DEBUG read example %r', example['video_matrix'][0, 0, :])
    if include_video_id:
      self.assertCountEqual(
          ['video_matrix', 'labels', 'num_frames', 'video_ids'], example.keys())
    else:
      self.assertCountEqual(['video_matrix', 'labels', 'num_frames'],
                            example.keys())

    # Check tensor values.
    expected_context = examples[0].context.feature[
        'VIDEO_EMBEDDING/context_feature/floats'].float_list.value
    expected_feature = examples[0].feature_lists.feature_list[
        'FEATURE/feature/floats'].feature[0].float_list.value
    expected_labels = examples[0].context.feature[
        params.label_field].int64_list.value
    if per_feature_l2_norm:
      expected_feature = tf.math.l2_normalize(expected_feature, axis=-1)
      expected_context = tf.math.l2_normalize(expected_context, axis=-1)
    self.assertAllEqual(expected_feature,
                        example['video_matrix'][0, 0, params.feature_sizes[0]:])
    self.assertAllEqual(expected_context,
                        example['video_matrix'][0, 0, :params.feature_sizes[0]])
    self.assertAllEqual(
        np.nonzero(example['labels'][0, :].numpy())[0], expected_labels)
    self.assertGreater(np.nonzero(example['labels'][0].numpy())[0].shape[0], 0)

    # Check tensor shape.
    batch_size = params.global_batch_size
    expected_num_frames = params.num_sample_frames or params.max_frames
    self.assertEqual(
        example['video_matrix'].shape.as_list(),
        [batch_size, expected_num_frames, sum(params.feature_sizes)],
    )
    self.assertEqual(example['labels'].shape.as_list(),
                     [batch_size, params.num_classes])
    self.assertEqual(example['num_frames'].shape.as_list(), [batch_size, 1])
    if include_video_id:
      self.assertEqual(example['video_ids'].shape.as_list(), [batch_size, 1])


if __name__ == '__main__':
  tf.test.main()
