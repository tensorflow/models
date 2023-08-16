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

"""Tests for action_localization data loader."""
import io
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

from official.projects.videoglue.datasets import action_localization

IMAGE_KEY = 'image/encoded'
KEYFRAME_INDEX = 'clip/key_frame/frame_index'
KEYFRAME_BOX_PREFIX = 'clip/key_frame/bbox'
DETECTED_BOX_PREFIX = 'centernet/bbox'
TFR_PATH = '/tmp/example.tfrecord'


def create_fake_tfse_sstable():
  """Creates fake data."""
  random_image = np.random.randint(0, 256, size=(263, 320, 3), dtype=np.uint8)
  random_image = Image.fromarray(random_image)
  with io.BytesIO() as buffer:
    random_image.save(buffer, format='JPEG')
    raw_image_bytes = buffer.getvalue()

  num_frames = 4
  tfse = tf.train.SequenceExample()
  # keyframe index
  tfse.context.feature.get_or_create(KEYFRAME_INDEX).int64_list.value[:] = [2]
  # keyframe boxes
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/ymin').float_list.value[:] = [0.0, 0.1, 0.2, 0.2]
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/xmin').float_list.value[:] = [0.0, 0.1, 0.2, 0.2]
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/ymax').float_list.value[:] = [0.5, 0.6, 0.7, 0.7]
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/xmax').float_list.value[:] = [0.5, 0.6, 0.7, 0.7]
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/score').float_list.value[:] = [1.0, 1.0, 1.0, 1.0]
  # boxes labels
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/label/index').int64_list.value[:] = [
          0, 1, 2, 3
      ]
  tfse.context.feature.get_or_create(
      f'{KEYFRAME_BOX_PREFIX}/label/string').bytes_list.value[:] = [
          b'0', b'1', b'2', b'3',
      ]

  for i in range(num_frames):
    # image
    tfse.feature_lists.feature_list.get_or_create(
        IMAGE_KEY).feature.add().bytes_list.value[:] = [raw_image_bytes]
    # detected boxes.
    tfse.feature_lists.feature_list.get_or_create(
        f'{DETECTED_BOX_PREFIX}/ymin').feature.add().float_list.value[:] = [
            0.0, 0.1, 0.2
        ]
    tfse.feature_lists.feature_list.get_or_create(
        f'{DETECTED_BOX_PREFIX}/xmin').feature.add().float_list.value[:] = [
            0.0, 0.1, 0.2
        ]
    tfse.feature_lists.feature_list.get_or_create(
        f'{DETECTED_BOX_PREFIX}/ymax').feature.add().float_list.value[:] = [
            0.5, 0.6, 0.7
        ]
    tfse.feature_lists.feature_list.get_or_create(
        f'{DETECTED_BOX_PREFIX}/xmax').feature.add().float_list.value[:] = [
            0.5, 0.6, 0.7
        ]
    tfse.feature_lists.feature_list.get_or_create(
        f'{DETECTED_BOX_PREFIX}/score').feature.add().float_list.value[:] = [
            0.91, 0.91, 0.1 * i
        ]

  writer = tf.io.TFRecordWriter(TFR_PATH)
  writer.write(tfse.SerializeToString())
  logging.info('Writing tfrecord table: %s', TFR_PATH)
  writer.close()


class ActionLocalizationTest(tf.test.TestCase):

  def test_create_action_localization_dataset(self):
    create_fake_tfse_sstable()
    dataset_cls = action_localization.ActionLocalizationBaseFactory(
        subset='train')
    dataset_cls._NUM_CLASSES = 5
    dataset_cls._ZERO_BASED_INDEX = True
    configs = {
        'is_training': False,
        'num_frames': 4,
        'temporal_stride': 1,
        'num_instance_per_frame': 5,
        'one_hot_label': True,
        'merge_multi_labels': True,
        'import_detected_bboxes': True,
        'augmentation_type': 'ava',
        'augmentation_params': {'scale_min': 0.0, 'scale_max': 0.0}
    }
    dataset_cls.configure(**configs)
    ds = dataset_cls.make_dataset(shuffle=False, batch_size=1)
    ds_iter = iter(ds)
    data = next(ds_iter)
    expected_subset = [
        'image',
        'keyframe_index',
        'label',
        'instances_position',
        'instances_mask',
        'instances_score',
        'nonmerge_label',
        'nonmerge_instances_position',
        'detected_instances_position',
        'detected_instances_mask',
        'detected_instances_score',
    ]
    self.assertSameElements(expected_subset, data.keys())

    self.assertAllEqual(data['keyframe_index'], [[2]])
    expected_label = tf.constant(
        [[1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0.],
         [0., 0., 1., 1., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]])
    expected_label = expected_label[None, ...]
    self.assertAllEqual(data['label'], expected_label)

    expected_instances_mask = tf.constant([True, True, True, False, False])
    expected_instances_mask = expected_instances_mask[None, :]
    self.assertAllEqual(data['instances_mask'], expected_instances_mask)

    expected_nonmerge_label = tf.constant([0, 1, 2, 3, -1])
    expected_nonmerge_label = expected_nonmerge_label[None, :]
    self.assertAllEqual(data['nonmerge_label'], expected_nonmerge_label)

    self.assertAllEqual(data['detected_instances_position'].shape, [1, 5, 4])
    self.assertAllEqual(data['detected_instances_mask'].shape, [1, 5])
    expected_detected_instances_mask = tf.constant(
        [[True, True, False, False, False]])
    self.assertAllEqual(data['detected_instances_mask'],
                        expected_detected_instances_mask)


if __name__ == '__main__':
  tf.test.main()
