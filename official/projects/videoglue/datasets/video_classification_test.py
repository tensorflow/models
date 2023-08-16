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

"""Tests for video_classification."""
import io
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

from official.projects.videoglue.datasets import video_classification
from official.vision.configs import common as common_cfg


IMAGE_KEY = 'image/encoded'
LABEL_KEY = 'clip/label/index'
TFR_PATH = '/tmp/sample.tfrecord'


def create_fake_tfse_sstable():
  """Creates fake data."""
  num_frames = 25
  tfse = tf.train.SequenceExample()
  tfse.context.feature.get_or_create(LABEL_KEY).int64_list.value[:] = [0]
  for frame_id in range(num_frames):
    image = np.ones((263, 320, 3), dtype=np.uint8) * frame_id
    image = Image.fromarray(image)
    with io.BytesIO() as buffer:
      image.save(buffer, format='JPEG')
      raw_image_bytes = buffer.getvalue()

    tfse.feature_lists.feature_list.get_or_create(
        IMAGE_KEY).feature.add().bytes_list.value[:] = [raw_image_bytes]

  writer = tf.io.TFRecordWriter(TFR_PATH)
  writer.write(tfse.SerializeToString())
  logging.info('Writing tfrecord table: %s', TFR_PATH)
  writer.close()


class VideoClassificationTest(tf.test.TestCase):

  def test_create_video_classification_data(self):
    create_fake_tfse_sstable()
    dataset_cls = video_classification.VideoClassificationBaseFactory(
        subset='train')
    configs = {
        'is_training': True,
        'num_frames': 4,
        'one_hot_label': True,
    }
    dataset_cls.configure(**configs)
    ds = dataset_cls.make_dataset(shuffle=False, batch_size=2)
    ds_iter = iter(ds)
    data = next(ds_iter)
    self.assertAllEqual(data['label'].shape, [2, 400])
    self.assertAllEqual(data['image'].shape, [2, 4, 224, 224, 3])

  def test_video_classification_randaug(self):
    create_fake_tfse_sstable()
    dataset_cls = video_classification.VideoClassificationBaseFactory(
        subset='train')
    configs = {
        'is_training': True,
        'num_frames': 4,
        'one_hot_label': True,
        'randaug_params': common_cfg.RandAugment().as_dict(),
    }
    dataset_cls.configure(**configs)
    ds = dataset_cls.make_dataset(shuffle=False, batch_size=2)
    ds_iter = iter(ds)
    data = next(ds_iter)
    self.assertAllEqual(data['label'].shape, [2, 400])
    self.assertAllEqual(data['image'].shape, [2, 4, 224, 224, 3])

  def test_video_classification_autoaug(self):
    create_fake_tfse_sstable()
    dataset_cls = video_classification.VideoClassificationBaseFactory(
        subset='train')
    configs = {
        'is_training': True,
        'num_frames': 4,
        'one_hot_label': True,
        'autoaug_params': common_cfg.AutoAugment().as_dict(),
    }
    dataset_cls.configure(**configs)
    ds = dataset_cls.make_dataset(shuffle=False, batch_size=2)
    ds_iter = iter(ds)
    data = next(ds_iter)
    self.assertAllEqual(data['label'].shape, [2, 400])
    self.assertAllEqual(data['image'].shape, [2, 4, 224, 224, 3])

  def test_video_classification_mixup_cutmix(self):
    create_fake_tfse_sstable()
    dataset_cls = video_classification.VideoClassificationBaseFactory(
        subset='train')
    configs = {
        'is_training': True,
        'num_frames': 4,
        'one_hot_label': True,
        'mixup_cutmix_params': common_cfg.MixupAndCutmix().as_dict(),
    }
    dataset_cls.configure(**configs)
    ds = dataset_cls.make_dataset(shuffle=False, batch_size=2)
    ds_iter = iter(ds)
    data = next(ds_iter)
    self.assertAllEqual(data['label'].shape, [2, 400])
    self.assertAllEqual(data['image'].shape, [2, 4, 224, 224, 3])

  def test_video_classification_sample_segments(self):
    create_fake_tfse_sstable()
    dataset_cls = video_classification.VideoClassificationBaseFactory(
        subset='train')
    configs = {
        'is_training': False,
        'num_frames': 5,
        'temporal_stride': 1,
        'sample_from_segments': True,
        'one_hot_label': True,
        'mixup_cutmix_params': common_cfg.MixupAndCutmix().as_dict(),
    }
    dataset_cls.configure(**configs)
    ds = dataset_cls.make_dataset(shuffle=False, batch_size=1)
    ds_iter = iter(ds)
    data = next(ds_iter)
    self.assertAllEqual(data['label'].shape, [1, 400])
    self.assertAllEqual(data['image'].shape, [1, 5, 224, 224, 3])
    average_image = tf.reduce_mean(data['image'] * 255., axis=[2, 3, 4])
    self.assertAllEqual(average_image[0].numpy(), [2.0, 7.0, 12.0, 16.0, 21.0])


if __name__ == '__main__':
  tf.test.main()
