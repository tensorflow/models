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
"""Tests for object_detection.utils.seq_example_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import seq_example_util
from object_detection.utils import tf_version


class SeqExampleUtilTest(tf.test.TestCase):

  def materialize_tensors(self, list_of_tensors):
    if tf_version.is_tf2():
      return [tensor.numpy() for tensor in list_of_tensors]
    else:
      with self.cached_session() as sess:
        return sess.run(list_of_tensors)

  def test_make_unlabeled_example(self):
    num_frames = 5
    image_height = 100
    image_width = 200
    dataset_name = b'unlabeled_dataset'
    video_id = b'video_000'
    images = tf.cast(tf.random.uniform(
        [num_frames, image_height, image_width, 3],
        maxval=256,
        dtype=tf.int32), dtype=tf.uint8)
    image_source_ids = [str(idx) for idx in range(num_frames)]
    images_list = tf.unstack(images, axis=0)
    encoded_images_list = [tf.io.encode_jpeg(image) for image in images_list]
    encoded_images = self.materialize_tensors(encoded_images_list)
    seq_example = seq_example_util.make_sequence_example(
        dataset_name=dataset_name,
        video_id=video_id,
        encoded_images=encoded_images,
        image_height=image_height,
        image_width=image_width,
        image_format='JPEG',
        image_source_ids=image_source_ids)

    context_feature_dict = seq_example.context.feature
    self.assertEqual(
        dataset_name,
        context_feature_dict['example/dataset_name'].bytes_list.value[0])
    self.assertEqual(
        0,
        context_feature_dict['clip/start/timestamp'].int64_list.value[0])
    self.assertEqual(
        num_frames - 1,
        context_feature_dict['clip/end/timestamp'].int64_list.value[0])
    self.assertEqual(
        num_frames,
        context_feature_dict['clip/frames'].int64_list.value[0])
    self.assertEqual(
        3,
        context_feature_dict['image/channels'].int64_list.value[0])
    self.assertEqual(
        b'JPEG',
        context_feature_dict['image/format'].bytes_list.value[0])
    self.assertEqual(
        image_height,
        context_feature_dict['image/height'].int64_list.value[0])
    self.assertEqual(
        image_width,
        context_feature_dict['image/width'].int64_list.value[0])
    self.assertEqual(
        video_id,
        context_feature_dict['clip/media_id'].bytes_list.value[0])

    seq_feature_dict = seq_example.feature_lists.feature_list
    self.assertLen(
        seq_feature_dict['image/encoded'].feature[:],
        num_frames)
    timestamps = [
        feature.int64_list.value[0] for feature
        in seq_feature_dict['image/timestamp'].feature]
    self.assertAllEqual(list(range(num_frames)), timestamps)
    source_ids = [
        feature.bytes_list.value[0] for feature
        in seq_feature_dict['image/source_id'].feature]
    self.assertAllEqual(
        [six.ensure_binary(str(idx)) for idx in range(num_frames)],
        source_ids)

  def test_make_labeled_example(self):
    num_frames = 3
    image_height = 100
    image_width = 200
    dataset_name = b'unlabeled_dataset'
    video_id = b'video_000'
    labels = [b'dog', b'cat', b'wolf']
    images = tf.cast(tf.random.uniform(
        [num_frames, image_height, image_width, 3],
        maxval=256,
        dtype=tf.int32), dtype=tf.uint8)
    images_list = tf.unstack(images, axis=0)
    encoded_images_list = [tf.io.encode_jpeg(image) for image in images_list]
    encoded_images = self.materialize_tensors(encoded_images_list)
    timestamps = [100000, 110000, 120000]
    is_annotated = [1, 0, 1]
    bboxes = [
        np.array([[0., 0., 0., 0.],
                  [0., 0., 1., 1.]], dtype=np.float32),
        np.zeros([0, 4], dtype=np.float32),
        np.array([], dtype=np.float32)
    ]
    label_strings = [
        np.array(labels),
        np.array([]),
        np.array([])
    ]

    seq_example = seq_example_util.make_sequence_example(
        dataset_name=dataset_name,
        video_id=video_id,
        encoded_images=encoded_images,
        image_height=image_height,
        image_width=image_width,
        timestamps=timestamps,
        is_annotated=is_annotated,
        bboxes=bboxes,
        label_strings=label_strings)

    context_feature_dict = seq_example.context.feature
    self.assertEqual(
        dataset_name,
        context_feature_dict['example/dataset_name'].bytes_list.value[0])
    self.assertEqual(
        timestamps[0],
        context_feature_dict['clip/start/timestamp'].int64_list.value[0])
    self.assertEqual(
        timestamps[-1],
        context_feature_dict['clip/end/timestamp'].int64_list.value[0])
    self.assertEqual(
        num_frames,
        context_feature_dict['clip/frames'].int64_list.value[0])

    seq_feature_dict = seq_example.feature_lists.feature_list
    self.assertLen(
        seq_feature_dict['image/encoded'].feature[:],
        num_frames)
    actual_timestamps = [
        feature.int64_list.value[0] for feature
        in seq_feature_dict['image/timestamp'].feature]
    self.assertAllEqual(timestamps, actual_timestamps)
    # Frame 0.
    self.assertAllEqual(
        is_annotated[0],
        seq_feature_dict['region/is_annotated'].feature[0].int64_list.value[0])
    self.assertAllClose(
        [0., 0.],
        seq_feature_dict['region/bbox/ymin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 0.],
        seq_feature_dict['region/bbox/xmin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 1.],
        seq_feature_dict['region/bbox/ymax'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 1.],
        seq_feature_dict['region/bbox/xmax'].feature[0].float_list.value[:])
    self.assertAllEqual(
        labels,
        seq_feature_dict['region/label/string'].feature[0].bytes_list.value[:])

    # Frame 1.
    self.assertAllEqual(
        is_annotated[1],
        seq_feature_dict['region/is_annotated'].feature[1].int64_list.value[0])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/ymin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/xmin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/ymax'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/xmax'].feature[1].float_list.value[:])
    self.assertAllEqual(
        [],
        seq_feature_dict['region/label/string'].feature[1].bytes_list.value[:])

  def test_make_labeled_example_with_context_features(self):
    num_frames = 2
    image_height = 100
    image_width = 200
    dataset_name = b'unlabeled_dataset'
    video_id = b'video_000'
    labels = [b'dog', b'cat']
    images = tf.cast(tf.random.uniform(
        [num_frames, image_height, image_width, 3],
        maxval=256,
        dtype=tf.int32), dtype=tf.uint8)
    images_list = tf.unstack(images, axis=0)
    encoded_images_list = [tf.io.encode_jpeg(image) for image in images_list]
    encoded_images = self.materialize_tensors(encoded_images_list)
    timestamps = [100000, 110000]
    is_annotated = [1, 0]
    bboxes = [
        np.array([[0., 0., 0., 0.],
                  [0., 0., 1., 1.]], dtype=np.float32),
        np.zeros([0, 4], dtype=np.float32)
    ]
    label_strings = [
        np.array(labels),
        np.array([])
    ]
    context_features = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    context_feature_length = [3]
    context_features_image_id_list = [b'im_1', b'im_2']

    seq_example = seq_example_util.make_sequence_example(
        dataset_name=dataset_name,
        video_id=video_id,
        encoded_images=encoded_images,
        image_height=image_height,
        image_width=image_width,
        timestamps=timestamps,
        is_annotated=is_annotated,
        bboxes=bboxes,
        label_strings=label_strings,
        context_features=context_features,
        context_feature_length=context_feature_length,
        context_features_image_id_list=context_features_image_id_list)

    context_feature_dict = seq_example.context.feature
    self.assertEqual(
        dataset_name,
        context_feature_dict['example/dataset_name'].bytes_list.value[0])
    self.assertEqual(
        timestamps[0],
        context_feature_dict['clip/start/timestamp'].int64_list.value[0])
    self.assertEqual(
        timestamps[-1],
        context_feature_dict['clip/end/timestamp'].int64_list.value[0])
    self.assertEqual(
        num_frames,
        context_feature_dict['clip/frames'].int64_list.value[0])

    self.assertAllClose(
        context_features,
        context_feature_dict['image/context_features'].float_list.value[:])
    self.assertEqual(
        context_feature_length[0],
        context_feature_dict[
            'image/context_feature_length'].int64_list.value[0])
    self.assertEqual(
        context_features_image_id_list,
        context_feature_dict[
            'image/context_features_image_id_list'].bytes_list.value[:])

    seq_feature_dict = seq_example.feature_lists.feature_list
    self.assertLen(
        seq_feature_dict['image/encoded'].feature[:],
        num_frames)
    actual_timestamps = [
        feature.int64_list.value[0] for feature
        in seq_feature_dict['image/timestamp'].feature]
    self.assertAllEqual(timestamps, actual_timestamps)
    # Frame 0.
    self.assertAllEqual(
        is_annotated[0],
        seq_feature_dict['region/is_annotated'].feature[0].int64_list.value[0])
    self.assertAllClose(
        [0., 0.],
        seq_feature_dict['region/bbox/ymin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 0.],
        seq_feature_dict['region/bbox/xmin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 1.],
        seq_feature_dict['region/bbox/ymax'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 1.],
        seq_feature_dict['region/bbox/xmax'].feature[0].float_list.value[:])
    self.assertAllEqual(
        labels,
        seq_feature_dict['region/label/string'].feature[0].bytes_list.value[:])

    # Frame 1.
    self.assertAllEqual(
        is_annotated[1],
        seq_feature_dict['region/is_annotated'].feature[1].int64_list.value[0])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/ymin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/xmin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/ymax'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict['region/bbox/xmax'].feature[1].float_list.value[:])
    self.assertAllEqual(
        [],
        seq_feature_dict['region/label/string'].feature[1].bytes_list.value[:])

  def test_make_labeled_example_with_predictions(self):
    num_frames = 2
    image_height = 100
    image_width = 200
    dataset_name = b'unlabeled_dataset'
    video_id = b'video_000'
    images = tf.cast(tf.random.uniform(
        [num_frames, image_height, image_width, 3],
        maxval=256,
        dtype=tf.int32), dtype=tf.uint8)
    images_list = tf.unstack(images, axis=0)
    encoded_images_list = [tf.io.encode_jpeg(image) for image in images_list]
    encoded_images = self.materialize_tensors(encoded_images_list)
    bboxes = [
        np.array([[0., 0., 0.75, 0.75],
                  [0., 0., 1., 1.]], dtype=np.float32),
        np.array([[0., 0.25, 0.5, 0.75]], dtype=np.float32)
    ]
    label_strings = [
        np.array(['cat', 'frog']),
        np.array(['cat'])
    ]
    detection_bboxes = [
        np.array([[0., 0., 0.75, 0.75]], dtype=np.float32),
        np.zeros([0, 4], dtype=np.float32)
    ]
    detection_classes = [
        np.array([5], dtype=np.int64),
        np.array([], dtype=np.int64)
    ]
    detection_scores = [
        np.array([0.9], dtype=np.float32),
        np.array([], dtype=np.float32)
    ]

    seq_example = seq_example_util.make_sequence_example(
        dataset_name=dataset_name,
        video_id=video_id,
        encoded_images=encoded_images,
        image_height=image_height,
        image_width=image_width,
        bboxes=bboxes,
        label_strings=label_strings,
        detection_bboxes=detection_bboxes,
        detection_classes=detection_classes,
        detection_scores=detection_scores)

    context_feature_dict = seq_example.context.feature
    self.assertEqual(
        dataset_name,
        context_feature_dict['example/dataset_name'].bytes_list.value[0])
    self.assertEqual(
        0,
        context_feature_dict['clip/start/timestamp'].int64_list.value[0])
    self.assertEqual(
        1,
        context_feature_dict['clip/end/timestamp'].int64_list.value[0])
    self.assertEqual(
        num_frames,
        context_feature_dict['clip/frames'].int64_list.value[0])

    seq_feature_dict = seq_example.feature_lists.feature_list
    self.assertLen(
        seq_feature_dict['image/encoded'].feature[:],
        num_frames)
    actual_timestamps = [
        feature.int64_list.value[0] for feature
        in seq_feature_dict['image/timestamp'].feature]
    self.assertAllEqual([0, 1], actual_timestamps)
    # Frame 0.
    self.assertAllEqual(
        1,
        seq_feature_dict['region/is_annotated'].feature[0].int64_list.value[0])
    self.assertAllClose(
        [0., 0.],
        seq_feature_dict['region/bbox/ymin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0., 0.],
        seq_feature_dict['region/bbox/xmin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.75, 1.],
        seq_feature_dict['region/bbox/ymax'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.75, 1.],
        seq_feature_dict['region/bbox/xmax'].feature[0].float_list.value[:])
    self.assertAllEqual(
        [b'cat', b'frog'],
        seq_feature_dict['region/label/string'].feature[0].bytes_list.value[:])
    self.assertAllClose(
        [0.],
        seq_feature_dict[
            'predicted/region/bbox/ymin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.],
        seq_feature_dict[
            'predicted/region/bbox/xmin'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.75],
        seq_feature_dict[
            'predicted/region/bbox/ymax'].feature[0].float_list.value[:])
    self.assertAllClose(
        [0.75],
        seq_feature_dict[
            'predicted/region/bbox/xmax'].feature[0].float_list.value[:])
    self.assertAllEqual(
        [5],
        seq_feature_dict[
            'predicted/region/label/index'].feature[0].int64_list.value[:])
    self.assertAllClose(
        [0.9],
        seq_feature_dict[
            'predicted/region/label/confidence'].feature[0].float_list.value[:])

    # Frame 1.
    self.assertAllEqual(
        1,
        seq_feature_dict['region/is_annotated'].feature[1].int64_list.value[0])
    self.assertAllClose(
        [0.0],
        seq_feature_dict['region/bbox/ymin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [0.25],
        seq_feature_dict['region/bbox/xmin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [0.5],
        seq_feature_dict['region/bbox/ymax'].feature[1].float_list.value[:])
    self.assertAllClose(
        [0.75],
        seq_feature_dict['region/bbox/xmax'].feature[1].float_list.value[:])
    self.assertAllEqual(
        [b'cat'],
        seq_feature_dict['region/label/string'].feature[1].bytes_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict[
            'predicted/region/bbox/ymin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict[
            'predicted/region/bbox/xmin'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict[
            'predicted/region/bbox/ymax'].feature[1].float_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict[
            'predicted/region/bbox/xmax'].feature[1].float_list.value[:])
    self.assertAllEqual(
        [],
        seq_feature_dict[
            'predicted/region/label/index'].feature[1].int64_list.value[:])
    self.assertAllClose(
        [],
        seq_feature_dict[
            'predicted/region/label/confidence'].feature[1].float_list.value[:])


if __name__ == '__main__':
  tf.test.main()
