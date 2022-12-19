# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow_models.official.projects.detr.dataloaders.coco."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from official.projects.detr.dataloaders import coco


def _gen_fn():
  h = np.random.randint(0, 300)
  w = np.random.randint(0, 300)
  num_boxes = np.random.randint(0, 50)
  return {
      'image': np.ones(shape=(h, w, 3), dtype=np.uint8),
      'image/id': np.random.randint(0, 100),
      'image/filename': 'test',
      'objects': {
          'is_crowd': np.ones(shape=(num_boxes), dtype=bool),
          'bbox': np.ones(shape=(num_boxes, 4), dtype=np.float32),
          'label': np.ones(shape=(num_boxes), dtype=np.int64),
          'id': np.ones(shape=(num_boxes), dtype=np.int64),
          'area': np.ones(shape=(num_boxes), dtype=np.int64),
      }
  }


class CocoDataloaderTest(tf.test.TestCase, parameterized.TestCase):

  def test_load_dataset(self):
    output_size = 1280
    max_num_boxes = 100
    batch_size = 2
    data_config = coco.COCODataConfig(
        tfds_name='coco/2017',
        tfds_split='validation',
        is_training=False,
        global_batch_size=batch_size,
        output_size=(output_size, output_size),
        max_num_boxes=max_num_boxes,
        )

    num_examples = 10
    def as_dataset(self, *args, **kwargs):
      del args
      del kwargs
      return tf.data.Dataset.from_generator(
          lambda: (_gen_fn() for i in range(num_examples)),
          output_types=self.info.features.dtype,
          output_shapes=self.info.features.shape,
      )

    with tfds.testing.mock_data(num_examples=num_examples,
                                as_dataset_fn=as_dataset):
      dataset = coco.COCODataLoader(data_config).load()
      dataset_iter = iter(dataset)
      images, labels = next(dataset_iter)
      self.assertEqual(images.shape, (batch_size, output_size, output_size, 3))
      self.assertEqual(labels['classes'].shape, (batch_size, max_num_boxes))
      self.assertEqual(labels['boxes'].shape, (batch_size, max_num_boxes, 4))
      self.assertEqual(labels['id'].shape, (batch_size,))
      self.assertEqual(
          labels['image_info'].shape, (batch_size, 4, 2))
      self.assertEqual(labels['is_crowd'].shape, (batch_size, max_num_boxes))

  @parameterized.named_parameters(
      ('training', True),
      ('validation', False))
  def test_preprocess(self, is_training):
    output_size = 1280
    max_num_boxes = 100
    batch_size = 2
    data_config = coco.COCODataConfig(
        tfds_name='coco/2017',
        tfds_split='validation',
        is_training=is_training,
        global_batch_size=batch_size,
        output_size=(output_size, output_size),
        max_num_boxes=max_num_boxes,
        )

    dl = coco.COCODataLoader(data_config)
    inputs = _gen_fn()
    image, label = dl.preprocess(inputs)
    self.assertEqual(image.shape, (output_size, output_size, 3))
    self.assertEqual(label['classes'].shape, (max_num_boxes))
    self.assertEqual(label['boxes'].shape, (max_num_boxes, 4))
    if not is_training:
      self.assertDTypeEqual(label['id'], int)
      self.assertEqual(
          label['image_info'].shape, (4, 2))
      self.assertEqual(label['is_crowd'].shape, (max_num_boxes))


if __name__ == '__main__':
  tf.test.main()
