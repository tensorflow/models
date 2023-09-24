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

"""Video ssl pretrain task definition."""
from typing import Any, Optional

from absl import logging
import tensorflow as tf
from official.core import input_reader
from official.core import task_factory
from official.projects.const_cl.configs import const_cl as exp_cfg
from official.projects.const_cl.datasets import video_ssl_inputs
from official.projects.const_cl.losses import losses
from official.projects.video_ssl.tasks import pretrain as video_ssl_pretrain
from official.vision.modeling import factory_3d


@task_factory.register_task_cls(exp_cfg.ConstCLPretrainTask)
class ConstCLPretrainTask(video_ssl_pretrain.VideoSSLPretrainTask):
  """A task for video contextualized ssl pretraining."""

  def build_model(self):
    """Builds video ssl pretraining model."""
    common_input_shape = [
        d1 if d1 == d2 else None
        for d1, d2 in zip(self.task_config.train_data.feature_shape,
                          self.task_config.validation_data.feature_shape)
    ]

    num_frames = common_input_shape[0]
    num_instances = self.task_config.train_data.num_instances
    input_specs_dict = {
        'image':
            tf.keras.layers.InputSpec(shape=[None] + common_input_shape),
        'instances_position':
            tf.keras.layers.InputSpec(
                shape=[None, num_frames, num_instances, 4]),
        'instances_mask':
            tf.keras.layers.InputSpec(shape=[None, num_frames, num_instances]),
    }

    logging.info('Build model input %r', common_input_shape)

    model = factory_3d.build_model(
        self.task_config.model.model_type,
        input_specs=input_specs_dict,
        model_config=self.task_config.model,
        num_classes=self.task_config.train_data.num_classes)
    return model

  def build_inputs(self,
                   params: exp_cfg.DataConfig,
                   input_context: Optional[Any] = None) -> tf.data.Dataset:
    """Builds ConST-CL SSL input."""

    parser = video_ssl_inputs.Parser(input_params=params)
    postprocess_fn = video_ssl_inputs.PostBatchProcessor(params)

    reader = input_reader.InputReader(
        params,
        dataset_fn=self._get_dataset_fn(params),
        decoder_fn=self._get_decoder_fn(params),
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=postprocess_fn)

    dataset = reader.read(input_context=input_context)
    return dataset

  def build_losses(self, model_outputs, num_replicas, model):
    """Sparse categorical cross entropy loss.

    Args:
      model_outputs: Output logits of the model.
      num_replicas: distributed replica number.
      model: keras model for calculating weight decay.

    Returns:
      The total loss tensor.
    """
    all_losses = {}
    logging_metrics = {}
    losses_config = self.task_config.losses
    total_loss = None

    global_loss = losses.ContrastiveLoss(
        normalize_inputs=losses_config.normalize_inputs,
        temperature=losses_config.global_temperature)
    local_loss = losses.InstanceContrastiveLoss(
        normalize_inputs=losses_config.normalize_inputs,
        temperature=losses_config.local_temperature)
    # Compute global loss.
    global_inputs = model_outputs['global_embeddings']
    global_loss_dict = global_loss(inputs=global_inputs,
                                   num_replicas=num_replicas)
    # Compute local loss.
    local_inputs = {
        'instances_a2b': model_outputs['inst_a2b'],
        'instances_b2a': model_outputs['inst_b2a'],
        'instances_a': model_outputs['inst_a'],
        'instances_b': model_outputs['inst_b'],
        'masks_a': model_outputs['masks_a'],
        'masks_b': model_outputs['masks_b'],
    }
    local_loss_dict = local_loss(predictions=local_inputs,
                                 num_replicas=num_replicas)
    # Compute regularization loss.
    reg_loss = losses_config.l2_weight_decay * tf.add_n([
        tf.nn.l2_loss(v) for v in model.trainable_variables
        if 'kernel' in v.name])

    total_loss = (global_loss_dict['loss'] * losses_config.global_weight +
                  local_loss_dict['loss'] * losses_config.local_weight +
                  reg_loss)
    all_losses.update({
        'total_loss': total_loss
    })
    all_losses[self.loss] = total_loss

    logging_metrics['regularization_loss'] = reg_loss
    for k, v in global_loss_dict.items():
      logging_metrics['global_loss/' + k] = v
    for k, v in local_loss_dict.items():
      logging_metrics['local_loss/' + k] = v
    return all_losses, logging_metrics

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation."""
    metrics = [
        tf.keras.metrics.Mean(name='regularization_loss'),

        tf.keras.metrics.Mean(name='global_loss/loss'),
        tf.keras.metrics.Mean(name='global_loss/contrastive_accuracy'),
        tf.keras.metrics.Mean(name='global_loss/contrastive_entropy'),

        tf.keras.metrics.Mean(name='local_loss/loss'),
        tf.keras.metrics.Mean(name='local_loss/positive_similarity_mean'),
        tf.keras.metrics.Mean(name='local_loss/positive_similarity_max'),
        tf.keras.metrics.Mean(name='local_loss/positive_similarity_min'),
        tf.keras.metrics.Mean(name='local_loss/negative_similarity_mean'),
        tf.keras.metrics.Mean(name='local_loss/negative_similarity_max'),
        tf.keras.metrics.Mean(name='local_loss/negative_similarity_min'),
    ]
    return metrics

  def process_metrics(self, metrics, contrastive_metrics):
    """Processes and updates metrics."""
    for metric in metrics:
      v = contrastive_metrics[metric.name]
      metric.update_state(v)

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Forward and backward pass.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, _ = inputs

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      all_losses, contrastive_metrics = self.build_losses(
          model_outputs=outputs, num_replicas=num_replicas,
          model=model)
      scaled_loss = all_losses[self.loss]

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = all_losses
    if metrics:
      self.process_metrics(metrics, contrastive_metrics)
      logs.update({m.name: m.result() for m in metrics})
    return logs

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    raise NotImplementedError

  def inference_step(self, features, model):
    """Performs the forward step."""
    raise NotImplementedError
