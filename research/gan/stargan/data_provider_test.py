"""Tests for google3.third_party.tensorflow_models.gan.stargan.data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google3.testing.pybase import googletest
import data_provider

mock = tf.test.mock


class DataProviderTest(googletest.TestCase):

  @mock.patch.object(
      data_provider.data_provider, 'provide_custom_data', autospec=True)
  def test_data_provider(self, mock_provide_custom_data):

    batch_size = 2
    patch_size = 8
    num_domains = 3

    images_shape = [batch_size, patch_size, patch_size, 3]
    mock_provide_custom_data.return_value = [
        tf.zeros(images_shape) for _ in range(num_domains)
    ]

    images, labels = data_provider.provide_data(
        image_file_patterns=None, batch_size=batch_size, patch_size=patch_size)

    self.assertEqual(num_domains, len(images))
    self.assertEqual(num_domains, len(labels))
    for label in labels:
      self.assertListEqual([batch_size, num_domains], label.shape.as_list())
    for image in images:
      self.assertListEqual(images_shape, image.shape.as_list())


if __name__ == '__main__':
  googletest.main()
