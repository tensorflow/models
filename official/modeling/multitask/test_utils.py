# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Testing utils for mock models and tasks."""
from typing import Dict, Text
import tensorflow as tf, tf_keras
from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling.multitask import base_model


class MockFooModel(tf_keras.Model):
  """A mock model can consume 'foo' and 'bar' inputs."""

  def __init__(self, shared_layer, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._share_layer = shared_layer
    self._foo_specific_layer = tf_keras.layers.Dense(1)
    self.inputs = {"foo": tf_keras.Input(shape=(2,), dtype=tf.float32),
                   "bar": tf_keras.Input(shape=(2,), dtype=tf.float32)}

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    self.add_loss(tf.zeros((1,), dtype=tf.float32))
    if "foo" in inputs:
      input_tensor = inputs["foo"]
    else:
      input_tensor = inputs["bar"]
    return self._foo_specific_layer(self._share_layer(input_tensor))


class MockBarModel(tf_keras.Model):
  """A mock model can only consume 'bar' inputs."""

  def __init__(self, shared_layer, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._share_layer = shared_layer
    self._bar_specific_layer = tf_keras.layers.Dense(1)
    self.inputs = {"bar": tf_keras.Input(shape=(2,), dtype=tf.float32)}

  def call(self, inputs):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    self.add_loss(tf.zeros((2,), dtype=tf.float32))
    return self._bar_specific_layer(self._share_layer(inputs["bar"]))


class MockMultiTaskModel(base_model.MultiTaskBaseModel):

  def __init__(self, *args, **kwargs):
    self._shared_dense = tf_keras.layers.Dense(1)
    super().__init__(*args, **kwargs)

  def _instantiate_sub_tasks(self) -> Dict[Text, tf_keras.Model]:
    return {
        "foo": MockFooModel(self._shared_dense),
        "bar": MockBarModel(self._shared_dense)
    }


def mock_data(feature_name):
  """Mock dataset function."""

  def _generate_data(_):
    x = tf.zeros(shape=(2,), dtype=tf.float32)
    label = tf.zeros([1], dtype=tf.int32)
    return {feature_name: x}, label

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      _generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset.prefetch(buffer_size=1).batch(2, drop_remainder=True)


class FooConfig(cfg.TaskConfig):
  pass


class BarConfig(cfg.TaskConfig):
  pass


@task_factory.register_task_cls(FooConfig)
class MockFooTask(base_task.Task):
  """Mock foo task object for testing."""

  def build_metrics(self, training: bool = True):
    del training
    return [tf_keras.metrics.Accuracy(name="foo_acc")]

  def build_inputs(self, params):
    return mock_data("foo")

  def build_model(self) -> tf_keras.Model:
    return MockFooModel(shared_layer=tf_keras.layers.Dense(1))

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    loss = tf_keras.losses.mean_squared_error(labels, model_outputs)
    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf.reduce_mean(loss)


@task_factory.register_task_cls(BarConfig)
class MockBarTask(base_task.Task):
  """Mock bar task object for testing."""

  def build_metrics(self, training: bool = True):
    del training
    return [tf_keras.metrics.Accuracy(name="bar_acc")]

  def build_inputs(self, params):
    return mock_data("bar")

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    loss = tf_keras.losses.mean_squared_error(labels, model_outputs)
    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf.reduce_mean(loss)
