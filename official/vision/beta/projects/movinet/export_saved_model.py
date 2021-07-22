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
  --export_path=/tmp/movinet/ \
  --model_id=a0 \
  --causal=True \
  --conv_type="3d" \
  --num_classes=600 \
  --use_positional_encoding=False \
  --checkpoint_path=""
```

To use an exported saved_model, refer to export_saved_model_test.py.
"""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.projects.movinet.modeling import movinet
from official.vision.beta.projects.movinet.modeling import movinet_model

flags.DEFINE_string(
    'export_path', '/tmp/movinet/',
    'Export path to save the saved_model file.')
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
flags.DEFINE_string(
    'se_type', '3d',
    '3d, 2d, or 2plus3d. 3d uses the default 3D spatiotemporal global average'
    'pooling for squeeze excitation. 2d uses 2D spatial global average pooling '
    'on each frame. 2plus3d concatenates both 3D and 2D global average '
    'pooling.')
flags.DEFINE_string(
    'activation', 'swish',
    'The main activation to use across layers.')
flags.DEFINE_string(
    'gating_activation', 'sigmoid',
    'The gating activation to use in squeeze-excitation layers.')
flags.DEFINE_bool(
    'use_positional_encoding', False,
    'Whether to use positional encoding (only applied when causal=True).')
flags.DEFINE_integer(
    'num_classes', 600, 'The number of classes for prediction.')
flags.DEFINE_integer(
    'batch_size', None,
    'The batch size of the input. Set to None for dynamic input.')
flags.DEFINE_integer(
    'num_frames', None,
    'The number of frames of the input. Set to None for dynamic input.')
flags.DEFINE_integer(
    'image_size', None,
    'The resolution of the input. Set to None for dynamic input.')
flags.DEFINE_string(
    'checkpoint_path', '',
    'Checkpoint path to load. Leave blank for default initialization.')

FLAGS = flags.FLAGS


def main(_) -> None:
  input_specs = tf.keras.layers.InputSpec(shape=[
      FLAGS.batch_size,
      FLAGS.num_frames,
      FLAGS.image_size,
      FLAGS.image_size,
      3,
  ])

  # Use dimensions of 1 except the channels to export faster,
  # since we only really need the last dimension to build and get the output
  # states. These dimensions will be set to `None` once the model is built.
  input_shape = [1 if s is None else s for s in input_specs.shape]

  backbone = movinet.Movinet(
      FLAGS.model_id,
      causal=FLAGS.causal,
      conv_type=FLAGS.conv_type,
      use_external_states=FLAGS.causal,
      input_specs=input_specs,
      activation=FLAGS.activation,
      gating_activation=FLAGS.gating_activation,
      se_type=FLAGS.se_type,
      use_positional_encoding=FLAGS.use_positional_encoding)
  model = movinet_model.MovinetClassifier(
      backbone,
      num_classes=FLAGS.num_classes,
      output_states=FLAGS.causal,
      input_specs=dict(image=input_specs))
  model.build(input_shape)

  # Compile model to generate some internal Keras variables.
  model.compile()

  if FLAGS.checkpoint_path:
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(FLAGS.checkpoint_path)
    status.assert_existing_objects_matched()

  if FLAGS.causal:
    # Call the model once to get the output states. Call again with `states`
    # input to ensure that the inputs with the `states` argument is built
    # with the full output state shapes.
    input_image = tf.ones(input_shape)
    _, states = model({**model.init_states(input_shape), 'image': input_image})
    _, states = model({**states, 'image': input_image})

    # Create a function to explicitly set the names of the outputs
    def predict(inputs):
      outputs, states = model(inputs)
      return {**states, 'logits': outputs}

    specs = {
        name: tf.TensorSpec(spec.shape, name=name, dtype=spec.dtype)
        for name, spec in model.initial_state_specs(
            input_specs.shape).items()
    }
    specs['image'] = tf.TensorSpec(
        input_specs.shape, dtype=model.dtype, name='image')

    predict_fn = tf.function(predict, jit_compile=True)
    predict_fn = predict_fn.get_concrete_function(specs)

    init_states_fn = tf.function(model.init_states, jit_compile=True)
    init_states_fn = init_states_fn.get_concrete_function(
        tf.TensorSpec([5], dtype=tf.int32))

    signatures = {'call': predict_fn, 'init_states': init_states_fn}

    tf.keras.models.save_model(
        model, FLAGS.export_path, signatures=signatures)
  else:
    _ = model(tf.ones(input_shape))
    tf.keras.models.save_model(model, FLAGS.export_path)

  print(' ----- Done. Saved Model is saved at {}'.format(FLAGS.export_path))


if __name__ == '__main__':
  app.run(main)
