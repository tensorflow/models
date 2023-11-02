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
from absl import logging
import tensorflow as tf, tf_keras

# pylint: disable=unused-import
from official.core import input_reader
from official.core import task_factory
from official.projects.video_ssl.configs import video_ssl as exp_cfg
from official.projects.video_ssl.dataloaders import video_ssl_input
from official.projects.video_ssl.losses import losses
from official.projects.video_ssl.modeling import video_ssl_model
from official.vision.modeling import factory_3d
from official.vision.tasks import video_classification
# pylint: enable=unused-import


@task_factory.register_task_cls(exp_cfg.VideoSSLPretrainTask)
class VideoSSLPretrainTask(video_classification.VideoClassificationTask):
  """A task for video ssl pretraining."""

  def build_model(self):
    """Builds video ssl pretraining model."""
    common_input_shape = [
        d1 if d1 == d2 else None
        for d1, d2 in zip(self.task_config.train_data.feature_shape,
                          self.task_config.validation_data.feature_shape)
    ]
    input_specs = tf_keras.layers.InputSpec(shape=[None] + common_input_shape)
    logging.info('Build model input %r', common_input_shape)

    model = factory_3d.build_model(
        self.task_config.model.model_type,
        input_specs=input_specs,
        model_config=self.task_config.model,
        num_classes=self.task_config.train_data.num_classes)
    return model

  def _get_decoder_fn(self, params):
    decoder = video_ssl_input.Decoder()
    return decoder.decode

  def build_inputs(self, params: exp_cfg.DataConfig, input_context=None):
    """Builds classification input."""

    parser = video_ssl_input.Parser(input_params=params)
    postprocess_fn = video_ssl_input.PostBatchProcessor(params)

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
    contrastive_metrics = {}
    losses_config = self.task_config.losses
    total_loss = None
    contrastive_loss_dict = losses.contrastive_loss(
        model_outputs, num_replicas, losses_config.normalize_hidden,
        losses_config.temperature, model,
        self.task_config.losses.l2_weight_decay)
    total_loss = contrastive_loss_dict['total_loss']
    all_losses.update({
        'total_loss': total_loss
    })
    all_losses[self.loss] = total_loss
    contrastive_metrics.update({
        'contrast_acc': contrastive_loss_dict['contrast_acc'],
        'contrast_entropy': contrastive_loss_dict['contrast_entropy'],
        'reg_loss': contrastive_loss_dict['reg_loss']
    })
    return all_losses, contrastive_metrics

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation."""
    metrics = [
        tf_keras.metrics.Mean(name='contrast_acc'),
        tf_keras.metrics.Mean(name='contrast_entropy'),
        tf_keras.metrics.Mean(name='reg_loss')
    ]
    return metrics

  def process_metrics(self, metrics, contrastive_metrics):
    """Process and update metrics."""
    contrastive_metric_values = contrastive_metrics.values()
    for metric, contrastive_metric_value in zip(metrics,
                                                contrastive_metric_values):
      metric.update_state(contrastive_metric_value)

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
    features, _ = inputs

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      if self.task_config.train_data.output_audio:
        outputs = model(features, training=True)
      else:
        outputs = model(features['image'], training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(
          lambda x: tf.cast(x, tf.float32), outputs)

      all_losses, contrastive_metrics = self.build_losses(
          model_outputs=outputs, num_replicas=num_replicas,
          model=model)
      loss = all_losses[self.loss]
      scaled_loss = loss

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(
          optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
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
