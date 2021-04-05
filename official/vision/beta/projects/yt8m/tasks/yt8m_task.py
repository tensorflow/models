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

"""Video classification task definition."""
from absl import logging
import tensorflow as tf

from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.modeling import tf_utils
from official.vision.beta.projects.yt8m.configs import yt8m as yt8m_cfg
from official.vision.beta.projects.yt8m.dataloaders import yt8m_input
from official.vision.beta.projects.yt8m.eval_utils import eval_util
from official.vision.beta.projects.yt8m.modeling import yt8m_model_utils as utils
from official.vision.beta.projects.yt8m.modeling.yt8m_model import YT8MModel


@task_factory.register_task_cls(yt8m_cfg.YT8MTask)
class YT8MTask(base_task.Task):
  """A task for video classification."""

  def build_model(self):
    """Builds model for YT8M Task."""
    train_cfg = self.task_config.train_data
    common_input_shape = [None, sum(train_cfg.feature_sizes)]

    # [batch_size x num_frames x num_features]
    input_specs = tf.keras.layers.InputSpec(shape=[None] + common_input_shape)
    logging.info('Build model input %r', common_input_shape)

    # Model configuration.
    model_config = self.task_config.model
    model = YT8MModel(
        input_params=model_config,
        input_specs=input_specs,
        num_frames=train_cfg.num_frames,
        num_classes=train_cfg.num_classes)
    return model

  def build_inputs(self, params: yt8m_cfg.DataConfig, input_context=None):
    """Builds input.

    Args:
      params: configuration for input data
      input_context: indicates information about the compute replicas and input
        pipelines

    Returns:
      dataset: dataset fetched from reader
    """

    decoder = yt8m_input.Decoder(input_params=params)
    decoder_fn = decoder.decode
    parser = yt8m_input.Parser(input_params=params)
    parser_fn = parser.parse_fn(params.is_training)
    postprocess = yt8m_input.PostBatchProcessor(input_params=params)
    postprocess_fn = postprocess.post_fn
    transform_batch = yt8m_input.TransformBatcher(input_params=params)
    batch_fn = transform_batch.batch_fn

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder_fn,
        parser_fn=parser_fn,
        postprocess_fn=postprocess_fn,
        transform_and_batch_fn=batch_fn)

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(self, labels, model_outputs, aux_losses=None):
    """Sigmoid Cross Entropy.

    Args:
      labels: tensor containing truth labels.
      model_outputs: output logits of the classifier.
      aux_losses: tensor containing auxiliarly loss tensors, i.e. `losses` in
        keras.Model.

    Returns:
      Tensors: The total loss, model loss tensors.
    """
    losses_config = self.task_config.losses
    model_loss = tf.keras.losses.binary_crossentropy(
        labels,
        model_outputs,
        from_logits=losses_config.from_logits,
        label_smoothing=losses_config.label_smoothing)

    model_loss = tf_utils.safe_mean(model_loss)
    total_loss = model_loss
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return total_loss, model_loss

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation.

       metric: mAP/gAP
       top_k: A positive integer specifying how many predictions are considered
        per video.
       top_n: A positive Integer specifying the average precision at n, or None
        to use all provided data points.
    Args:
      training: bool value, true for training mode, false for eval/validation.

    Returns:
      list of strings that indicate metrics to be used
    """
    metrics = []
    metric_names = ['total_loss', 'model_loss']
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    if not training:  # Cannot run in train step.
      num_classes = self.task_config.validation_data.num_classes
      top_k = self.task_config.top_k
      top_n = self.task_config.top_n
      self.avg_prec_metric = eval_util.EvaluationMetrics(
          num_classes, top_k=top_k, top_n=top_n)

    return metrics

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors. output_dict = {
          "video_ids": batch_video_ids,
          "video_matrix": batch_video_matrix,
          "labels": batch_labels,
          "num_frames": batch_frames, }
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      a dictionary of logs.
    """
    features, labels = inputs['video_matrix'], inputs['labels']
    num_frames = inputs['num_frames']

    # Normalize input features.
    feature_dim = len(features.shape) - 1
    features = tf.nn.l2_normalize(features, feature_dim)

    # sample random frames / random sequence
    num_frames = tf.cast(num_frames, tf.float32)
    sample_frames = self.task_config.train_data.num_frames
    if self.task_config.model.sample_random_frames:
      features = utils.SampleRandomFrames(features, num_frames, sample_frames)
    else:
      features = utils.SampleRandomSequence(features, num_frames, sample_frames)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)

      # Computes per-replica loss
      loss, model_loss = self.build_losses(
          model_outputs=outputs, labels=labels, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer,
                    tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer,
                  tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)

    # Apply gradient clipping.
    if self.task_config.gradient_clip_norm > 0:
      grads, _ = tf.clip_by_global_norm(grads,
                                        self.task_config.gradient_clip_norm)
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}

    all_losses = {'total_loss': loss, 'model_loss': model_loss}

    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
        logs.update({m.name: m.result()})

    return logs

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors. output_dict = {
        "video_ids": batch_video_ids,
        "video_matrix": batch_video_matrix,
        "labels": batch_labels,
        "num_frames": batch_frames, }
      model: the model, forward definition
      metrics: a nested structure of metrics objects.

    Returns:
      a dictionary of logs.
    """
    features, labels = inputs['video_matrix'], inputs['labels']
    num_frames = inputs['num_frames']

    # Normalize input features.
    feature_dim = len(features.shape) - 1
    features = tf.nn.l2_normalize(features, feature_dim)

    # sample random frames (None, 5, 1152) -> (None, 30, 1152)
    sample_frames = self.task_config.validation_data.num_frames
    if self.task_config.model.sample_random_frames:
      features = utils.SampleRandomFrames(features, num_frames, sample_frames)
    else:
      features = utils.SampleRandomSequence(features, num_frames, sample_frames)

    outputs = self.inference_step(features, model)
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    if self.task_config.validation_data.segment_labels:
      # workaround to ignore the unrated labels.
      outputs *= inputs['label_weights']
      # remove padding
      outputs = outputs[~tf.reduce_all(labels == -1, axis=1)]
      labels = labels[~tf.reduce_all(labels == -1, axis=1)]
    loss, model_loss = self.build_losses(
        model_outputs=outputs, labels=labels, aux_losses=model.losses)

    logs = {self.loss: loss}

    all_losses = {'total_loss': loss, 'model_loss': model_loss}

    logs.update({self.avg_prec_metric.name: (labels, outputs)})

    if metrics:
      for m in metrics:
        m.update_state(all_losses[m.name])
        logs.update({m.name: m.result()})
    return logs

  def inference_step(self, inputs, model):
    """Performs the forward step."""
    return model(inputs, training=False)

  def aggregate_logs(self, state=None, step_logs=None):
    if state is None:
      state = self.avg_prec_metric
    self.avg_prec_metric.accumulate(
        labels=step_logs[self.avg_prec_metric.name][0],
        predictions=step_logs[self.avg_prec_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    avg_prec_metrics = self.avg_prec_metric.get()
    self.avg_prec_metric.clear()
    return avg_prec_metrics
