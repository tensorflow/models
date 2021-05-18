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

# Lint as: python3
r"""Exports models to tf.saved_model.

Export example:

```shell
python3 export_saved_model.py \
  --output_path=/tmp/movinet/ \
  --model_id=a0 \
  --causal=True \
  --conv_type="3d" \
  --num_classes=600 \
  --checkpoint_path=""
```

To use an exported saved_model in various applications:

```python
import tensorflow as tf
import tensorflow_hub as hub

saved_model_path = ...

inputs = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32)

encoder = hub.KerasLayer(saved_model_path, trainable=True)
outputs = encoder(inputs)

model = tf.keras.Model(inputs, outputs)

example_input = tf.ones([1, 8, 172, 172, 3])
outputs = model(example_input, states)
```
"""

from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.projects.movinet.modeling import movinet
from official.vision.beta.projects.movinet.modeling import movinet_model

flags.DEFINE_string(
    'output_path', '/tmp/movinet/',
    'Path to saved exported saved_model file.')
flags.DEFINE_string(
    'model_id', 'a0', 'MoViNet model name.')
flags.DEFINE_bool(
    'causal', False, 'Run the model in causal mode.')
flags.DEFINE_string(
    'conv_type', '3d',
    '3d, 2plus1d, or 3d_2plus1d. 3d configures the network '
    'to use the default 3D convolution. 2plus1d uses (2+1)D convolution '
    'with Conv2D operations and 2D reshaping (e.g., a 5x3x3 kernel becomes '
    '3x3 followed by 5x1 conv). 3d_2plus1d uses (2+1)D convolution with '
    'Conv3D and no 2D reshaping (e.g., a 5x3x3 kernel becomes 1x3x3 '
    'followed by 5x1x1 conv).')
flags.DEFINE_integer(
    'num_classes', 600, 'The number of classes for prediction.')
flags.DEFINE_string(
    'checkpoint_path', '',
    'Checkpoint path to load. Leave blank for default initialization.')

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Use dimensions of 1 except the channels to export faster,
  # since we only really need the last dimension to build and get the output
  # states. These dimensions will be set to `None` once the model is built.
  input_shape = [1, 1, 1, 1, 3]

  backbone = movinet.Movinet(
      FLAGS.model_id, causal=FLAGS.causal, conv_type=FLAGS.conv_type)
  model = movinet_model.MovinetClassifier(
      backbone, num_classes=FLAGS.num_classes, output_states=FLAGS.causal)
  model.build(input_shape)

  if FLAGS.checkpoint_path:
    model.load_weights(FLAGS.checkpoint_path)

  if FLAGS.causal:
    # Call the model once to get the output states. Call again with `states`
    # input to ensure that the inputs with the `states` argument is built
    _, states = model(dict(image=tf.ones(input_shape), states={}))
    _, states = model(dict(image=tf.ones(input_shape), states=states))

    input_spec = tf.TensorSpec(
        shape=[None, None, None, None, 3],
        dtype=tf.float32,
        name='inputs')

    state_specs = {}
    for name, state in states.items():
      shape = state.shape
      if len(state.shape) == 5:
        shape = [None, state.shape[1], None, None, state.shape[-1]]
      new_spec = tf.TensorSpec(shape=shape, dtype=state.dtype, name=name)
      state_specs[name] = new_spec

    specs = (input_spec, state_specs)

    # Define a tf.keras.Model with custom signatures to allow it to accept
    # a state dict as an argument. We define it inline here because
    # we first need to determine the shape of the state tensors before
    # applying the `input_signature` argument to `tf.function`.
    class ExportStateModule(tf.Module):
      """Module with state for exporting to saved_model."""

      def __init__(self, model):
        self.model = model

      @tf.function(input_signature=[input_spec])
      def __call__(self, inputs):
        return self.model(dict(image=inputs, states={}))

      @tf.function(input_signature=[input_spec])
      def base(self, inputs):
        return self.model(dict(image=inputs, states={}))

      @tf.function(input_signature=specs)
      def stream(self, inputs, states):
        return self.model(dict(image=inputs, states=states))

    module = ExportStateModule(model)

    tf.saved_model.save(module, FLAGS.output_path)
  else:
    _ = model(tf.ones(input_shape))
    tf.keras.models.save_model(model, FLAGS.output_path)

  print(' ----- Done. Saved Model is saved at {}'.format(FLAGS.output_path))


if __name__ == '__main__':
  app.run(main)
