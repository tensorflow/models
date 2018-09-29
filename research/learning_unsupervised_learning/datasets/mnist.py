# Copyright 2018 Google, Inc. All Rights Reserved.
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


import sonnet as snt
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from learning_unsupervised_learning.datasets import common

class Mnist(snt.AbstractModule):
  def __init__(self, device, batch_size=128, name="Mnist"):
    self.device = device
    self.batch_size = batch_size

    self._make_dataset()
    self.iterator = None

    super(Mnist, self).__init__(name=name)

  def _make_dataset(self):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(self.batch_size * 3)
    dataset = dataset.batch(self.batch_size)
    def _map_fn(image, label):
      image = tf.to_float(image) / 255.
      label.set_shape([self.batch_size])
      label = tf.cast(label, dtype=tf.int32)
      label_onehot = tf.one_hot(label, 10)
      image = tf.reshape(image, [self.batch_size, 28, 28, 1])
      return common.ImageLabelOnehot(
          image=image, label=label, label_onehot=label_onehot)

    self.dataset = dataset.map(_map_fn)

  def _build(self):
    if self.iterator is None:
      self.iterator = self.dataset.make_one_shot_iterator()
    batch = self.iterator.get_next()
    [b.set_shape([self.batch_size] + b.shape.as_list()[1:]) for b in batch]
    return batch


class TinyMnist(Mnist):
  def __init__(self, *args, **kwargs):
    kwargs.setdefault("name", "TinyMnist")
    super(TinyMnist, self).__init__(*args, **kwargs)

  def _make_dataset(self):
    super(TinyMnist, self)._make_dataset()

    def _map_fn(batch):
      new_img = tf.image.resize_images(batch.image, [14, 14])
      return common.ImageLabelOnehot(
          image=new_img, label=batch.label, label_onehot=batch.label_onehot)

    self.dataset = self.dataset.map(_map_fn)
