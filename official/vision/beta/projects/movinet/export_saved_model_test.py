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

"""Tests for export_saved_model."""

from absl import flags
import tensorflow as tf
import tensorflow_hub as hub

from official.vision.beta.projects.movinet import export_saved_model

FLAGS = flags.FLAGS


class ExportSavedModelTest(tf.test.TestCase):

  def test_movinet_export_a0_base_with_tfhub(self):
    saved_model_path = self.get_temp_dir()

    FLAGS.export_path = saved_model_path
    FLAGS.model_id = 'a0'
    FLAGS.causal = False
    FLAGS.num_classes = 600

    export_saved_model.main('unused_args')

    encoder = hub.KerasLayer(saved_model_path, trainable=True)

    inputs = tf.keras.layers.Input(
        shape=[None, None, None, 3],
        dtype=tf.float32)

    outputs = encoder(dict(image=inputs))

    model = tf.keras.Model(inputs, outputs)

    example_input = tf.ones([1, 8, 172, 172, 3])
    outputs = model(example_input)

    self.assertEqual(outputs.shape, [1, 600])

  def test_movinet_export_a0_stream_with_tfhub(self):
    saved_model_path = self.get_temp_dir()

    FLAGS.export_path = saved_model_path
    FLAGS.model_id = 'a0'
    FLAGS.causal = True
    FLAGS.num_classes = 600

    export_saved_model.main('unused_args')

    encoder = hub.KerasLayer(saved_model_path, trainable=True)

    image_input = tf.keras.layers.Input(
        shape=[None, None, None, 3],
        dtype=tf.float32,
        name='image')

    init_states_fn = encoder.resolved_object.signatures['init_states']
    state_shapes = {
        name: ([s if s > 0 else None for s in state.shape], state.dtype)
        for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
    }
    states_input = {
        name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
        for name, (shape, dtype) in state_shapes.items()
    }

    inputs = {**states_input, 'image': image_input}

    outputs = encoder(inputs)

    model = tf.keras.Model(inputs, outputs)

    example_input = tf.ones([1, 8, 172, 172, 3])
    frames = tf.split(example_input, example_input.shape[1], axis=1)

    init_states = init_states_fn(tf.shape(example_input))

    expected_outputs, _ = model({**init_states, 'image': example_input})

    states = init_states
    for frame in frames:
      outputs, states = model({**states, 'image': frame})

    self.assertEqual(outputs.shape, [1, 600])
    self.assertNotEmpty(states)
    self.assertAllClose(outputs, expected_outputs, 1e-5, 1e-5)

if __name__ == '__main__':
  tf.test.main()
