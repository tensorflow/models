# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.core.export_base_v2."""
import os

import tensorflow as tf, tf_keras

from official.core import export_base
from official.vision.serving import export_base_v2


class TestModel(tf_keras.Model):

  def __init__(self):
    super().__init__()
    self._dense = tf_keras.layers.Dense(2)

  def call(self, inputs):
    return {'outputs': self._dense(inputs)}


class ExportBaseTest(tf.test.TestCase):

  def test_preprocessor(self):
    tmp_dir = self.get_temp_dir()
    model = TestModel()
    inputs = tf.ones([2, 4], tf.float32)

    preprocess_fn = lambda inputs: 2 * inputs

    module = export_base_v2.ExportModule(
        params=None,
        input_signature=tf.TensorSpec(shape=[2, 4]),
        model=model,
        preprocessor=preprocess_fn)
    expected_output = model(preprocess_fn(inputs))
    ckpt_path = tf.train.Checkpoint(model=model).save(
        os.path.join(tmp_dir, 'ckpt'))
    export_dir = export_base.export(
        module, ['serving_default'],
        export_savedmodel_dir=tmp_dir,
        checkpoint_path=ckpt_path,
        timestamped=False)
    imported = tf.saved_model.load(export_dir)
    output = imported.signatures['serving_default'](inputs)
    print('output', output)
    self.assertAllClose(
        output['outputs'].numpy(), expected_output['outputs'].numpy())

  def test_postprocessor(self):
    tmp_dir = self.get_temp_dir()
    model = TestModel()
    inputs = tf.ones([2, 4], tf.float32)

    postprocess_fn = lambda logits: {'outputs': 2 * logits['outputs']}

    module = export_base_v2.ExportModule(
        params=None,
        model=model,
        input_signature=tf.TensorSpec(shape=[2, 4]),
        postprocessor=postprocess_fn)
    expected_output = postprocess_fn(model(inputs))
    ckpt_path = tf.train.Checkpoint(model=model).save(
        os.path.join(tmp_dir, 'ckpt'))
    export_dir = export_base.export(
        module, ['serving_default'],
        export_savedmodel_dir=tmp_dir,
        checkpoint_path=ckpt_path,
        timestamped=False)
    imported = tf.saved_model.load(export_dir)
    output = imported.signatures['serving_default'](inputs)
    self.assertAllClose(
        output['outputs'].numpy(), expected_output['outputs'].numpy())


if __name__ == '__main__':
  tf.test.main()
