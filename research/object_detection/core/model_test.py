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

"""Tests for model API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from object_detection.core import model
from object_detection.utils import test_case


class FakeModel(model.DetectionModel):

  def __init__(self):

    # sub-networks containing weights of different shapes.
    self._network1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 1)
    ])

    self._network2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 1)
    ])

    super(FakeModel, self).__init__(num_classes=0)

  def preprocess(self, images):
    return images, tf.shape(images)

  def predict(self, images, shapes):
    return {'prediction': self._network2(self._network1(images))}

  def postprocess(self, prediction_dict, shapes):
    return prediction_dict

  def loss(self):
    return tf.constant(0.0)

  def updates(self):
    return []

  def restore_map(self):
    return {}

  def restore_from_objects(self, fine_tune_checkpoint_type):
    pass

  def regularization_losses(self):
    return []


class ModelTest(test_case.TestCase):

  def test_model_call(self):

    detection_model = FakeModel()

    def graph_fn():
      return detection_model(tf.zeros((1, 128, 128, 3)))

    result = self.execute(graph_fn, [])
    self.assertEqual(result['prediction'].shape,
                     (1, 128, 128, 16))

  def test_freeze(self):

    detection_model = FakeModel()
    detection_model(tf.zeros((1, 128, 128, 3)))

    net1_var_shapes = [tuple(var.get_shape().as_list()) for var in
                       detection_model._network1.trainable_variables]

    del detection_model

    detection_model = FakeModel()
    detection_model._network2.trainable = False
    detection_model(tf.zeros((1, 128, 128, 3)))

    var_shapes = [tuple(var.get_shape().as_list()) for var in
                  detection_model._network1.trainable_variables]

    self.assertEqual(set(net1_var_shapes), set(var_shapes))


if __name__ == '__main__':
  tf.test.main()
