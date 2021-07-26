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

"""An example task definition for image classification."""
from typing import Any, List, Optional, Tuple, Sequence, Mapping

import tensorflow as tf

from official.common import dataset_fn
from official.core import base_task
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.beta.dataloaders import input_reader_factory
from official.vision.beta.projects.example import example_config as exp_cfg
from official.vision.beta.projects.example import example_input
from official.vision.beta.projects.example import example_model


@task_factory.register_task_cls(exp_cfg.ExampleTask)
class ExampleTask(base_task.Task):
  """Class of an example task.

  A task is a subclass of base_task.Task that defines model, input, loss, metric
  and one training and evaluation step, etc.
  """

  def build_model(self) -> tf.keras.Model:
    """Builds a model."""
    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            self.task_config.model.input_size)

    model = example_model.build_example_model(
        input_specs=input_specs, model_config=self.task_config.model)
    return model

  def build_inputs(
      self,
      params: exp_cfg.ExampleDataConfig,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Builds input.

    The input from this function is a tf.data.Dataset that has gone through
    pre-processing steps, such as augmentation, batching, shuffuling, etc.

    Args:
      params: The experiment config.
      input_context: An optional InputContext used by input reader.

    Returns:
      A tf.data.Dataset object.
    """

    num_classes = self.task_config.model.num_classes
    input_size = self.task_config.model.input_size
    decoder = example_input.Decoder()

    parser = example_input.Parser(
        output_size=input_size[:2], num_classes=num_classes)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self,
                   labels: tf.Tensor,
                   model_outputs: tf.Tensor,
                   aux_losses: Optional[Any] = None) -> tf.Tensor:
    """Builds losses for training and validation.

    Args:
      labels: Input groundtruth labels.
      model_outputs: Output of the model.
      aux_losses: The auxiliarly loss tensors, i.e. `losses` in tf.keras.Model.

    Returns:
      The total loss tensor.
    """
    total_loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, model_outputs, from_logits=True)
    total_loss = tf_utils.safe_mean(total_loss)

    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss

  def build_metrics(self,
                    training: bool = True) -> Sequence[tf.keras.metrics.Metric]:
    """Gets streaming metrics for training/validation.

    This function builds and returns a list of metrics to compute during
    training and validation. The list contains objects of subclasses of
    tf.keras.metrics.Metric. Training and validation can have different metrics.

    Args:
      training: Whether the metric is for training or not.

    Returns:
     A list of tf.keras.metrics.Metric objects.
    """
    k = self.task_config.evaluation.top_k
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=k, name='top_{}_accuracy'.format(k))
    ]
    return metrics

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:
    """Does forward and backward.

    This example assumes input is a tuple of (features, labels), which follows
    the output from data loader, i.e., Parser. The output from Parser is fed
    into train_step to perform one step forward and backward pass. Other data
    structure, such as dictionary, can also be used, as long as it is consistent
    between output from Parser and input used here.

    Args:
      inputs: A tuple of of input tensors of (features, labels).
      model: A tf.keras.Model instance.
      optimizer: The optimizer for this training step.
      metrics: A nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

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
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    return logs

  def validation_step(self,
                      inputs: Tuple[Any, Any],
                      model: tf.keras.Model,
                      metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:
    """Runs validatation step.

    Args:
      inputs: A tuple of of input tensors of (features, labels).
      model: A tf.keras.Model instance.
      metrics: A nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    loss = self.build_losses(
        model_outputs=outputs, labels=labels, aux_losses=model.losses)

    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    return logs

  def inference_step(self, inputs: tf.Tensor, model: tf.keras.Model) -> Any:
    """Performs the forward step. It is used in validation_step."""
    return model(inputs, training=False)
