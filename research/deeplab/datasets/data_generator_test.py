# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import range
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
        self.assertEqual(batch[common.HEIGHT][0], image_attributes.height)
        self.assertEqual(batch[common.WIDTH][0], image_attributes.width)
        self.assertEqual(batch[common.IMAGE_NAME][0],
                         image_attributes.image_name.encode())

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
        image=None,
        label=None,
        height=366,
        width=500,
        image_name='2007_000033')
  elif index == 1:
    return ImageAttributes(
        image=None,
        label=None,
        height=335,
        width=500,
        image_name='2007_000042')
  elif index == 2:
    return ImageAttributes(
        image=None,
        label=None,
        height=333,
        width=500,
        image_name='2007_000061')
  else:
    raise ValueError('Index can only be 0, 1 or 2.')


if __name__ == '__main__':
  tf.test.main()
