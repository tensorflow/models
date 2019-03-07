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
"""Tests for deeplab.datasets.data_generator."""

from __future__ import print_function

import collections

import tensorflow as tf

from deeplab import common
from deeplab.datasets import data_generator

ImageAttributes = collections.namedtuple(
    'ImageAttributes', ['image', 'label', 'height', 'width', 'image_name'])


class DatasetTest(tf.test.TestCase):

  # Note: training dataset cannot be tested since there is shuffle operation.
  # When disabling the shuffle, training dataset is operated same as validation
  # dataset. Therefore it is not tested again.
  def testPascalVocSegTestData(self):
    dataset = data_generator.Dataset(
        dataset_name='pascal_voc_seg',
        split_name='val',
        dataset_dir=
        'deeplab/testing/pascal_voc_seg',
        batch_size=1,
        crop_size=[3, 3],  # Use small size for testing.
        min_resize_value=3,
        max_resize_value=3,
        resize_factor=None,
        min_scale_factor=0.01,
        max_scale_factor=2.0,
        scale_factor_step_size=0.25,
        is_training=False,
        model_variant='mobilenet_v2')

    self.assertAllEqual(dataset.num_of_classes, 21)
    self.assertAllEqual(dataset.ignore_label, 255)

    num_of_images = 3
    with self.test_session() as sess:
      iterator = dataset.get_one_shot_iterator()

      for i in range(num_of_images):
        batch = iterator.get_next()
        batch, = sess.run([batch])
        image_attributes = _get_attributes_of_image(i)

        self.assertAllEqual(batch[common.IMAGE][0], image_attributes.image)
        self.assertAllEqual(batch[common.LABEL][0], image_attributes.label)
        self.assertEqual(batch[common.HEIGHT][0], image_attributes.height)
        self.assertEqual(batch[common.WIDTH][0], image_attributes.width)
        self.assertEqual(batch[common.IMAGE_NAME][0],
                         image_attributes.image_name)

      # All data have been read.
      with self.assertRaisesRegexp(tf.errors.OutOfRangeError, ''):
        sess.run([iterator.get_next()])


def _get_attributes_of_image(index):
  """Gets the attributes of the image.

  Args:
    index: Index of image in all images.

  Returns:
    Attributes of the image in the format of ImageAttributes.

  Raises:
    ValueError: If index is of wrong value.
  """
  if index == 0:
    return ImageAttributes(
        image=IMAGE_1,
        label=LABEL_1,
        height=366,
        width=500,
        image_name='2007_000033')
  elif index == 1:
    return ImageAttributes(
        image=IMAGE_2,
        label=LABEL_2,
        height=335,
        width=500,
        image_name='2007_000042')
  elif index == 2:
    return ImageAttributes(
        image=IMAGE_3,
        label=LABEL_3,
        height=333,
        width=500,
        image_name='2007_000061')
  else:
    raise ValueError('Index can only be 0, 1 or 2.')


IMAGE_1 = (
    (
        (57., 41., 18.),
        (151.5, 138., 111.5),
        (107., 158., 143.),
    ),
    (
        (104.5, 141., 191.),
        (101.75, 72.5, 120.75),
        (86.5, 139.5, 120.),
    ),
    (
        (96., 85., 145.),
        (123.5, 107.5, 97.),
        (61., 148., 116.),
    ),
)

LABEL_1 = (
    (
        (70,),
        (227,),
        (251,),
    ),
    (
        (101,),
        (0,),
        (10,),
    ),
    (
        (145,),
        (245,),
        (146,),
    ),
)

IMAGE_2 = (
    (
        (94., 64., 98.),
        (145.5, 136.5, 134.5),
        (108., 162., 172.),
    ),
    (
        (168., 157., 213.),
        (161.5, 154.5, 148.),
        (25., 46., 93.),
    ),
    (
        (255., 204., 237.),
        (124., 102., 126.5),
        (155., 181., 82.),
    ),
)

LABEL_2 = (
    (
        (44,),
        (146,),
        (121,),
    ),
    (
        (108,),
        (118,),
        (6,),
    ),
    (
        (246,),
        (121,),
        (108,),
    ),
)

IMAGE_3 = (
    (
        (235., 173., 150.),
        (145.5, 83.5, 102.),
        (82., 149., 158.),
    ),
    (
        (130., 95., 14.),
        (132.5, 141.5, 93.),
        (119., 85., 86.),
    ),
    (
        (127.5, 127.5, 127.5),
        (127.5, 127.5, 127.5),
        (127.5, 127.5, 127.5),
    ),
)

LABEL_3 = (
    (
        (91,),
        (120,),
        (132,),
    ),
    (
        (135,),
        (139,),
        (72,),
    ),
    (
        (255,),
        (255,),
        (255,),
    ),
)

if __name__ == '__main__':
  tf.test.main()
