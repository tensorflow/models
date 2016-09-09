# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow_models.im2txt.ops.image_embedding."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from im2txt.ops import image_embedding


class InceptionV3Test(tf.test.TestCase):

  def setUp(self):
    super(InceptionV3Test, self).setUp()

    batch_size = 4
    height = 299
    width = 299
    num_channels = 3
    self._images = tf.placeholder(tf.float32,
                                  [batch_size, height, width, num_channels])
    self._batch_size = batch_size

  def _countInceptionParameters(self):
    """Counts the number of parameters in the inception model at top scope."""
    counter = {}
    for v in tf.all_variables():
      name_tokens = v.op.name.split("/")
      if name_tokens[0] == "InceptionV3":
        name = "InceptionV3/" + name_tokens[1]
        num_params = v.get_shape().num_elements()
        assert num_params
        counter[name] = counter.get(name, 0) + num_params
    return counter

  def _verifyParameterCounts(self):
    """Verifies the number of parameters in the inception model."""
    param_counts = self._countInceptionParameters()
    expected_param_counts = {
        "InceptionV3/Conv2d_1a_3x3": 960,
        "InceptionV3/Conv2d_2a_3x3": 9312,
        "InceptionV3/Conv2d_2b_3x3": 18624,
        "InceptionV3/Conv2d_3b_1x1": 5360,
        "InceptionV3/Conv2d_4a_3x3": 138816,
        "InceptionV3/Mixed_5b": 256368,
        "InceptionV3/Mixed_5c": 277968,
        "InceptionV3/Mixed_5d": 285648,
        "InceptionV3/Mixed_6a": 1153920,
        "InceptionV3/Mixed_6b": 1298944,
        "InceptionV3/Mixed_6c": 1692736,
        "InceptionV3/Mixed_6d": 1692736,
        "InceptionV3/Mixed_6e": 2143872,
        "InceptionV3/Mixed_7a": 1699584,
        "InceptionV3/Mixed_7b": 5047872,
        "InceptionV3/Mixed_7c": 6080064,
    }
    self.assertDictEqual(expected_param_counts, param_counts)

  def _assertCollectionSize(self, expected_size, collection):
    actual_size = len(tf.get_collection(collection))
    if expected_size != actual_size:
      self.fail("Found %d items in collection %s (expected %d)." %
                (actual_size, collection, expected_size))

  def testTrainableTrueIsTrainingTrue(self):
    embeddings = image_embedding.inception_v3(
        self._images, trainable=True, is_training=True)
    self.assertEqual([self._batch_size, 2048], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(376, tf.GraphKeys.VARIABLES)
    self._assertCollectionSize(188, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(188, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(94, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(23, tf.GraphKeys.SUMMARIES)

  def testTrainableTrueIsTrainingFalse(self):
    embeddings = image_embedding.inception_v3(
        self._images, trainable=True, is_training=False)
    self.assertEqual([self._batch_size, 2048], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(376, tf.GraphKeys.VARIABLES)
    self._assertCollectionSize(188, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(94, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(23, tf.GraphKeys.SUMMARIES)

  def testTrainableFalseIsTrainingTrue(self):
    embeddings = image_embedding.inception_v3(
        self._images, trainable=False, is_training=True)
    self.assertEqual([self._batch_size, 2048], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(376, tf.GraphKeys.VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(23, tf.GraphKeys.SUMMARIES)

  def testTrainableFalseIsTrainingFalse(self):
    embeddings = image_embedding.inception_v3(
        self._images, trainable=False, is_training=False)
    self.assertEqual([self._batch_size, 2048], embeddings.get_shape().as_list())

    self._verifyParameterCounts()
    self._assertCollectionSize(376, tf.GraphKeys.VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.TRAINABLE_VARIABLES)
    self._assertCollectionSize(0, tf.GraphKeys.UPDATE_OPS)
    self._assertCollectionSize(0, tf.GraphKeys.REGULARIZATION_LOSSES)
    self._assertCollectionSize(0, tf.GraphKeys.LOSSES)
    self._assertCollectionSize(23, tf.GraphKeys.SUMMARIES)


if __name__ == "__main__":
  tf.test.main()
