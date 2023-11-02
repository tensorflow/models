# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

Export for TF Lite example:

```shell
python3 export_saved_model.py \
  --model_id=a0 \
  --causal=True \
  --conv_type=2plus1d \
  --se_type=2plus3d \
  --activation=hard_swish \
  --gating_activation=hard_sigmoid \
  --use_positional_encoding=False \
  --num_classes=600 \
  --batch_size=1 \
  --num_frames=1 \  # Use a single frame for streaming mode
  --image_size=172 \  # Input resolution for the model
  --bundle_input_init_states_fn=False \
  --checkpoint_path=/path/to/checkpoint \
  --export_path=/tmp/movinet_a0_stream
```

To use an exported saved_model, refer to export_saved_model_test.py.
"""

from typing import Optional, Tuple

from absl import app
from absl import flags
import tensorflow as tf, tf_keras

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

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
    'classifier_activation', 'swish',
    'The classifier activation to use.')
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
flags.DEFINE_bool(
    'bundle_input_init_states_fn', True,
    'Add init_states as a function signature to the saved model.'
    'This is not necessary if the input shape is static (e.g., for TF Lite).')
flags.DEFINE_string(
    'checkpoint_path', '',
    'Checkpoint path to load. Leave blank for default initialization.')
flags.DEFINE_bool(
    'assert_checkpoint_objects_matched',
    True,
    'Whether to check the checkpoint objects exactly match those of the model.',
)

FLAGS = flags.FLAGS


def export_saved_model(
    model: tf_keras.Model,
    input_shape: Tuple[int, int, int, int, int],
    export_path: str = '/tmp/movinet/',
    causal: bool = False,
    bundle_input_init_states_fn: bool = True,
    checkpoint_path: Optional[str] = None,
    assert_checkpoint_objects_matched: bool = True,
) -> None:
  """Exports a MoViNet model to a saved model.

  Args:
    model: the tf_keras.Model to export.
    input_shape: The 5D spatiotemporal input shape of size [batch_size,
      num_frames, image_height, image_width, num_channels]. Set the field or a
      shape position in the field to None for dynamic input.
    export_path: Export path to save the saved_model file.
    causal: Run the model in causal mode.
    bundle_input_init_states_fn: Add init_states as a function signature to the
      saved model. This is not necessary if the input shape is static (e.g., for
      TF Lite).
    checkpoint_path: Checkpoint path to load. Leave blank to keep the model's
      initialization.
    assert_checkpoint_objects_matched: Whether to check the checkpoint objects
      exactly match those of the model.
  """

  # Use dimensions of 1 except the channels to export faster,
  # since we only really need the last dimension to build and get the output
  # states. These dimensions can be set to `None` once the model is built.
  input_shape_concrete = [1 if s is None else s for s in input_shape]
  model.build(input_shape_concrete)

  # Compile model to generate some internal Keras variables.
  model.compile()

  if checkpoint_path:
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    if assert_checkpoint_objects_matched:
      status.assert_existing_objects_matched()

  if causal:
    # Call the model once to get the output states. Call again with `states`
    # input to ensure that the inputs with the `states` argument is built
    # with the full output state shapes.
    input_image = tf.ones(input_shape_concrete)
    _, states = model({
        **model.init_states(input_shape_concrete), 'image': input_image})
    _ = model({**states, 'image': input_image})

    # Create a function to explicitly set the names of the outputs
    def predict(inputs):
      outputs, states = model(inputs)
      return {**states, 'logits': outputs}

    specs = {
        name: tf.TensorSpec(spec.shape, name=name, dtype=spec.dtype)
        for name, spec in model.initial_state_specs(
            input_shape).items()
    }
    specs['image'] = tf.TensorSpec(
        input_shape, dtype=model.dtype, name='image')

    predict_fn = tf.function(predict, jit_compile=True)
    predict_fn = predict_fn.get_concrete_function(specs)

    init_states_fn = tf.function(model.init_states, jit_compile=True)
    init_states_fn = init_states_fn.get_concrete_function(
        tf.TensorSpec([5], dtype=tf.int32))

    if bundle_input_init_states_fn:
      signatures = {'call': predict_fn, 'init_states': init_states_fn}
    else:
      signatures = predict_fn

    tf_keras.models.save_model(
        model, export_path, signatures=signatures)
  else:
    _ = model(tf.ones(input_shape_concrete))
    tf_keras.models.save_model(model, export_path)


def build_and_export_saved_model(
    export_path: str = '/tmp/movinet/',
    model_id: str = 'a0',
    causal: bool = False,
    conv_type: str = '3d',
    se_type: str = '3d',
    activation: str = 'swish',
    classifier_activation: str = 'swish',
    gating_activation: str = 'sigmoid',
    use_positional_encoding: bool = False,
    num_classes: int = 600,
    input_shape: Optional[Tuple[int, int, int, int, int]] = None,
    bundle_input_init_states_fn: bool = True,
    checkpoint_path: Optional[str] = None,
    assert_checkpoint_objects_matched: bool = True,
) -> None:
  """Builds and exports a MoViNet model to a saved model.

  Args:
    export_path: Export path to save the saved_model file.
    model_id: MoViNet model name.
    causal: Run the model in causal mode.
    conv_type: 3d, 2plus1d, or 3d_2plus1d. 3d configures the network to use the
      default 3D convolution. 2plus1d uses (2+1)D convolution with Conv2D
      operations and 2D reshaping (e.g., a 5x3x3 kernel becomes 3x3 followed by
      5x1 conv). 3d_2plus1d uses (2+1)D convolution with Conv3D and no 2D
      reshaping (e.g., a 5x3x3 kernel becomes 1x3x3 followed by 5x1x1 conv).
    se_type: 3d, 2d, or 2plus3d. 3d uses the default 3D spatiotemporal global
      average pooling for squeeze excitation. 2d uses 2D spatial global average
      pooling on each frame. 2plus3d concatenates both 3D and 2D global average
      pooling.
    activation: The main activation to use across layers.
    classifier_activation: The classifier activation to use.
    gating_activation: The gating activation to use in squeeze-excitation
      layers.
    use_positional_encoding: Whether to use positional encoding (only applied
      when causal=True).
    num_classes: The number of classes for prediction.
    input_shape: The 5D spatiotemporal input shape of size [batch_size,
      num_frames, image_height, image_width, num_channels]. Set the field or a
      shape position in the field to None for dynamic input.
    bundle_input_init_states_fn: Add init_states as a function signature to the
      saved model. This is not necessary if the input shape is static (e.g., for
      TF Lite).
    checkpoint_path: Checkpoint path to load. Leave blank for default
      initialization.
    assert_checkpoint_objects_matched: Whether to check the checkpoint objects
      exactly match those of the model.
  """

  input_specs = tf_keras.layers.InputSpec(shape=input_shape)

  # Override swish activation implementation to remove custom gradients
  if activation == 'swish':
    activation = 'simple_swish'
  if classifier_activation == 'swish':
    classifier_activation = 'simple_swish'

  backbone = movinet.Movinet(
      model_id=model_id,
      causal=causal,
      use_positional_encoding=use_positional_encoding,
      conv_type=conv_type,
      se_type=se_type,
      input_specs=input_specs,
      activation=activation,
      gating_activation=gating_activation,
      use_sync_bn=False,
      use_external_states=causal)
  model = movinet_model.MovinetClassifier(
      backbone,
      num_classes=num_classes,
      output_states=causal,
      input_specs=dict(image=input_specs),
      activation=classifier_activation)

  export_saved_model(
      model=model,
      input_shape=input_shape,
      export_path=export_path,
      causal=causal,
      bundle_input_init_states_fn=bundle_input_init_states_fn,
      checkpoint_path=checkpoint_path,
      assert_checkpoint_objects_matched=assert_checkpoint_objects_matched,
  )


def main(_) -> None:
  input_shape = (
      FLAGS.batch_size, FLAGS.num_frames, FLAGS.image_size, FLAGS.image_size, 3)
  build_and_export_saved_model(
      export_path=FLAGS.export_path,
      model_id=FLAGS.model_id,
      causal=FLAGS.causal,
      conv_type=FLAGS.conv_type,
      se_type=FLAGS.se_type,
      activation=FLAGS.activation,
      classifier_activation=FLAGS.classifier_activation,
      gating_activation=FLAGS.gating_activation,
      use_positional_encoding=FLAGS.use_positional_encoding,
      num_classes=FLAGS.num_classes,
      input_shape=input_shape,
      bundle_input_init_states_fn=FLAGS.bundle_input_init_states_fn,
      checkpoint_path=FLAGS.checkpoint_path,
      assert_checkpoint_objects_matched=FLAGS.assert_checkpoint_objects_matched,
  )
  print(' ----- Done. Saved Model is saved at {}'.format(FLAGS.export_path))


if __name__ == '__main__':
  app.run(main)
