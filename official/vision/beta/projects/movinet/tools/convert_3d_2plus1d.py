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

"""Converts '3d_2plus1d' checkpoints into '2plus1d'."""

from absl import app
from absl import flags
import tensorflow as tf

from official.vision.beta.projects.movinet.modeling import movinet
from official.vision.beta.projects.movinet.modeling import movinet_model

flags.DEFINE_string(
    'input_checkpoint_path', None,
    'Checkpoint path to load.')
flags.DEFINE_string(
    'output_checkpoint_path', None,
    'Export path to save the saved_model file.')
flags.DEFINE_string(
    'model_id', 'a0', 'MoViNet model name.')
flags.DEFINE_bool(
    'causal', False, 'Run the model in causal mode.')
flags.DEFINE_bool(
    'use_positional_encoding', False,
    'Whether to use positional encoding (only applied when causal=True).')
flags.DEFINE_integer(
    'num_classes', 600, 'The number of classes for prediction.')
flags.DEFINE_bool(
    'verify_output', False, 'Verify the output matches between the models.')

FLAGS = flags.FLAGS


def main(_) -> None:
  backbone_2plus1d = movinet.Movinet(
      model_id=FLAGS.model_id,
      causal=FLAGS.causal,
      conv_type='2plus1d',
      use_positional_encoding=FLAGS.use_positional_encoding)
  model_2plus1d = movinet_model.MovinetClassifier(
      backbone=backbone_2plus1d,
      num_classes=FLAGS.num_classes)
  model_2plus1d.build([1, 1, 1, 1, 3])

  backbone_3d_2plus1d = movinet.Movinet(
      model_id=FLAGS.model_id,
      causal=FLAGS.causal,
      conv_type='3d_2plus1d',
      use_positional_encoding=FLAGS.use_positional_encoding)
  model_3d_2plus1d = movinet_model.MovinetClassifier(
      backbone=backbone_3d_2plus1d,
      num_classes=FLAGS.num_classes)
  model_3d_2plus1d.build([1, 1, 1, 1, 3])

  checkpoint = tf.train.Checkpoint(model=model_3d_2plus1d)
  status = checkpoint.restore(FLAGS.input_checkpoint_path)
  status.assert_existing_objects_matched()

  # Ensure both models have the same weights
  weights = []
  for var_2plus1d, var_3d_2plus1d in zip(
      model_2plus1d.get_weights(), model_3d_2plus1d.get_weights()):
    if var_2plus1d.shape == var_3d_2plus1d.shape:
      weights.append(var_3d_2plus1d)
    else:
      if var_3d_2plus1d.shape[0] == 1:
        weight = var_3d_2plus1d[0]
      else:
        weight = var_3d_2plus1d[:, 0]
      if weight.shape[-1] != var_2plus1d.shape[-1]:
        # Transpose any depthwise kernels (conv3d --> depthwise_conv2d)
        weight = tf.transpose(weight, perm=(0, 1, 3, 2))
      weights.append(weight)
  model_2plus1d.set_weights(weights)

  if FLAGS.verify_output:
    inputs = tf.random.uniform([1, 6, 64, 64, 3], dtype=tf.float32)

    logits_2plus1d = model_2plus1d(inputs)
    logits_3d_2plus1d = model_3d_2plus1d(inputs)

    if tf.reduce_mean(logits_2plus1d - logits_3d_2plus1d) > 1e-5:
      raise ValueError('Bad conversion, model outputs do not match.')

  save_checkpoint = tf.train.Checkpoint(
      model=model_2plus1d, backbone=backbone_2plus1d)
  save_checkpoint.save(FLAGS.output_checkpoint_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_checkpoint_path')
  flags.mark_flag_as_required('output_checkpoint_path')
  app.run(main)
