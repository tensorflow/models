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

"""Text classification task with ViT."""

import dataclasses
from typing import Tuple

import numpy as np
from scipy import stats
from sklearn import metrics as sklearn_metrics
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.data import data_loader_factory
from official.projects.pixel.modeling import pixel


@dataclasses.dataclass
class PixelModelConfig(base_config.Config):
  """The model configuration."""

  filters: int = 768
  num_layers: int = 12
  mlp_dim: int = 3072
  num_heads: int = 12
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  init_stochastic_depth_rate: float = 0.0


@dataclasses.dataclass
class PixelConfig(cfg.TaskConfig):
  """The task configuration."""

  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()
  patch_h: int = 16
  patch_w: int = 16
  num_classes: int = 2
  num_channels: int = 3
  input_size: Tuple[int, int] = (16, 4096)
  model: PixelModelConfig = PixelModelConfig()


@task_factory.register_task_cls(PixelConfig)
class PixelClassificationTask(base_task.Task):
  """Text classificaiton with Pixel and load checkpoint if exists."""

  label_field: str = 'label'
  metric_type: str = 'accuracy'

  def build_model(self) -> tf_keras.Model:
    encoder = pixel.VisionTransformer(
        self.task_config.patch_h,
        self.task_config.patch_w,
        self.task_config.model.filters,
        self.task_config.model.num_layers,
        self.task_config.model.mlp_dim,
        self.task_config.model.num_heads,
        self.task_config.model.dropout_rate,
        self.task_config.model.attention_dropout_rate,
        self.task_config.model.init_stochastic_depth_rate,
    )
    model = pixel.PixelLinearClassifier(
        encoder, self.task_config.num_classes, self.task_config.model.filters
    )
    h, w = self.task_config.input_size
    positions = h // self.task_config.patch_h * w // self.task_config.patch_w
    model({
        'label': tf.zeros((1,)),
        'pixel_values': tf.zeros((1, self.task_config.num_channels, h, w)),
        'attention_mask': tf.zeros((1, positions)),
    })
    return model

  def build_inputs(self, params, input_context=None):
    return data_loader_factory.get_data_loader(params).load(input_context)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    label_ids = labels[self.label_field]
    if self.task_config.num_classes == 1:
      loss = tf_keras.losses.mean_squared_error(label_ids, model_outputs)
    else:
      loss = tf_keras.losses.sparse_categorical_crossentropy(
          label_ids, tf.cast(model_outputs, tf.float32), from_logits=True
      )

    if aux_losses:
      loss += tf.add_n(aux_losses)
    return tf_utils.safe_mean(loss)

  def initialize(self, model: tf_keras.Model):
    """Load encoder if checkpoint exists.

    Args:
      model: The keras.Model built or used by this task.
    """
    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    ckpt = tf.train.Checkpoint(encoder=model.encoder)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()

  def build_metrics(self, training=None):
    del training
    if self.task_config.num_classes == 1:
      metrics = [tf_keras.metrics.MeanSquaredError()]
    elif self.task_config.num_classes == 2:
      metrics = [
          tf_keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
          tf_keras.metrics.AUC(name='auc', curve='PR'),
      ]
    else:
      metrics = [
          tf_keras.metrics.SparseCategoricalAccuracy(name='cls_accuracy'),
      ]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    for metric in metrics:
      if metric.name == 'auc':
        # Convert the logit to probability and extract the probability of True..
        metric.update_state(
            labels[self.label_field],
            tf.expand_dims(tf.nn.softmax(model_outputs)[:, 1], axis=1),
        )
      if metric.name == 'cls_accuracy':
        metric.update_state(labels[self.label_field], model_outputs)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    compiled_metrics.update_state(labels[self.label_field], model_outputs)

  def validation_step(self, inputs, model: tf_keras.Model, metrics=None):
    features, labels = inputs, inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses
    )
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    if self.metric_type == 'matthews_corrcoef':
      logs.update({
          'sentence_prediction': (
              tf.expand_dims(  # Ensure one prediction along batch dimension.
                  tf.math.argmax(outputs, axis=1), axis=1
              )
          ),
          'labels': labels[self.label_field],
      })
    else:
      logs.update({
          'sentence_prediction': outputs,
          'labels': labels[self.label_field],
      })
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if self.metric_type == 'accuracy':
      return None
    if state is None:
      state = {'sentence_prediction': [], 'labels': []}
    state['sentence_prediction'].append(
        np.concatenate(
            [v.numpy() for v in step_outputs['sentence_prediction']], axis=0
        )
    )
    state['labels'].append(
        np.concatenate([v.numpy() for v in step_outputs['labels']], axis=0)
    )
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    if self.metric_type == 'accuracy':
      return None

    preds = np.concatenate(aggregated_logs['sentence_prediction'], axis=0)
    labels = np.concatenate(aggregated_logs['labels'], axis=0)
    if self.metric_type == 'f1':
      preds = np.argmax(preds, axis=1)
      return {self.metric_type: sklearn_metrics.f1_score(labels, preds)}
    elif self.metric_type == 'matthews_corrcoef':
      preds = np.reshape(preds, -1)
      labels = np.reshape(labels, -1)
      return {
          self.metric_type: sklearn_metrics.matthews_corrcoef(preds, labels)
      }
    elif self.metric_type == 'pearson_spearman_corr':
      preds = np.reshape(preds, -1)
      labels = np.reshape(labels, -1)
      pearson_corr = stats.pearsonr(preds, labels)[0]
      spearman_corr = stats.spearmanr(preds, labels)[0]
      corr_metric = (pearson_corr + spearman_corr) / 2
      return {self.metric_type: corr_metric}
