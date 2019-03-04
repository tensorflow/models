# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.mnist import mnist
from official.mnist import mnist_eager


def device():
  return "/device:GPU:0" if tf.test.is_gpu_available() else "/device:CPU:0"


def data_format():
  return "channels_first" if tf.test.is_gpu_available() else "channels_last"


def random_dataset():
  batch_size = 64
  images = tf.random.normal([batch_size, 784])
  labels = tf.random.uniform([batch_size], minval=0, maxval=10, dtype=tf.int32)
  return tf.data.Dataset.from_tensors((images, labels))


def train(defun=False):
  model = mnist.create_model(data_format())
  #if defun:
  #  model.call = tfe.defun(model.call)
  optimizer = tf.optimizers.SGD(learning_rate=0.01)
  dataset = random_dataset()
  with tf.device(device()):
    mnist_eager.train(model, optimizer, dataset)


def evaluate(defun=False):
  model = mnist.create_model(data_format())
  dataset = random_dataset()
  #if defun:
  #  model.call = tfe.defun(model.call)
  with tf.device(device()):
    mnist_eager.test(model, dataset)


class MNISTTest(tf.test.TestCase):
  """Run tests for MNIST eager loop."""

  def test_train(self):
    train(defun=False)

  def test_evaluate(self):
    evaluate(defun=False)

  def test_train_with_defun(self):
    train(defun=True)

  def test_evaluate_with_defun(self):
    evaluate(defun=True)


if __name__ == "__main__":
  tf.test.main()
