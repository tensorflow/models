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
"""Image classification task definition."""
import tensorflow as tf

from official.core import input_reader
from official.core import task_factory
from official.vision.beta.dataloaders import classification_input
from official.vision.beta.projects.yolo.configs import darknet_classification as exp_cfg
from official.vision.beta.projects.yolo.dataloaders import classification_tfds_decoder as cli
from official.vision.beta.tasks import image_classification


@task_factory.register_task_cls(exp_cfg.ImageClassificationTask)
class ImageClassificationTask(image_classification.ImageClassificationTask):
  """A task for image classification."""

  def build_inputs(self, params, input_context=None):
    """Builds classification input."""

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size

    if params.tfds_name:
      decoder = cli.Decoder()
    else:
      decoder = classification_input.Decoder()

    parser = classification_input.Parser(
        output_size=input_size[:2],
        num_classes=num_classes,
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)
    return dataset

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    if self.task_config.losses.one_hot:
      labels = tf.one_hot(labels, self.task_config.model.num_classes)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss.
      loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)

    # Apply gradient clipping.
    if self.task_config.gradient_clip_norm > 0:
      grads, _ = tf.clip_by_global_norm(
          grads, self.task_config.gradient_clip_norm)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics})
    elif model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in model.metrics})
    return logs


