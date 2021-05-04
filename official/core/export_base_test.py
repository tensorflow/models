# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.core.export_base."""
import os
from typing import Any, Dict, Mapping, Text

import tensorflow as tf

from official.core import export_base


class TestModule(export_base.ExportModule):

  @tf.function
  def serve(self, inputs: tf.Tensor) -> Mapping[Text, tf.Tensor]:
    return {'outputs': self.inference_step(inputs)}

  def get_inference_signatures(
      self, function_keys: Dict[Text, Text]) -> Mapping[Text, Any]:
    input_signature = tf.TensorSpec(shape=[None, None], dtype=tf.float32)
    return {'foo': self.serve.get_concrete_function(input_signature)}


class ExportBaseTest(tf.test.TestCase):

  def test_export_module(self):
    tmp_dir = self.get_temp_dir()
    model = tf.keras.layers.Dense(2)
    inputs = tf.ones([2, 4], tf.float32)
    expected_output = model(inputs, training=False)
    module = TestModule(params=None, model=model)
    ckpt_path = tf.train.Checkpoint(model=model).save(
        os.path.join(tmp_dir, 'ckpt'))
    export_dir = export_base.export(
        module, ['foo'],
        export_savedmodel_dir=tmp_dir,
        checkpoint_path=ckpt_path,
        timestamped=True)
    self.assertTrue(os.path.exists(os.path.join(export_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(
            os.path.join(export_dir, 'variables', 'variables.index')))
    self.assertTrue(
        os.path.exists(
            os.path.join(export_dir, 'variables',
                         'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(export_dir)
    output = imported.signatures['foo'](inputs)
    self.assertAllClose(output['outputs'].numpy(), expected_output.numpy())

  def test_custom_inference_step(self):
    tmp_dir = self.get_temp_dir()
    model = tf.keras.layers.Dense(2)
    inputs = tf.ones([2, 4], tf.float32)

    def _inference_step(inputs, model):
      return tf.nn.softmax(model(inputs, training=False))

    module = TestModule(
        params=None, model=model, inference_step=_inference_step)
    expected_output = _inference_step(inputs, model)
    ckpt_path = tf.train.Checkpoint(model=model).save(
        os.path.join(tmp_dir, 'ckpt'))
    export_dir = export_base.export(
        module, ['foo'],
        export_savedmodel_dir=tmp_dir,
        checkpoint_path=ckpt_path,
        timestamped=False)
    imported = tf.saved_model.load(export_dir)
    output = imported.signatures['foo'](inputs)
    self.assertAllClose(output['outputs'].numpy(), expected_output.numpy())


if __name__ == '__main__':
  tf.test.main()
