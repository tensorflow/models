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

"""Video classification task definition."""
from typing import Dict, List, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import task_factory
from official.modeling import tf_utils
from official.projects.yt8m.configs import yt8m as yt8m_cfg
from official.projects.yt8m.dataloaders import yt8m_input
from official.projects.yt8m.eval_utils import eval_util
from official.projects.yt8m.modeling import yt8m_model
from official.core import input_reader


@task_factory.register_task_cls(yt8m_cfg.YT8MTask)
class YT8MTask(base_task.Task):
  """A task for video classification."""

  def build_model(self):
    """Builds model for YT8M Task."""
    train_cfg = self.task_config.train_data
    common_input_shape = [None, sum(train_cfg.feature_sizes)]

    # [batch_size x num_frames x num_features]
    input_specs = tf_keras.layers.InputSpec(shape=[None] + common_input_shape)
    logging.info('Build model input %r', common_input_shape)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Model configuration.
    model_config = self.task_config.model
    model = yt8m_model.VideoClassificationModel(
        params=model_config,
        input_specs=input_specs,
        num_classes=train_cfg.num_classes,
        l2_weight_decay=l2_weight_decay,
    )

    # Warmup calls to build model variables.
    _ = model(
        inputs=tf_keras.Input(common_input_shape, dtype=tf.float32),
        num_frames=tf_keras.Input([], dtype=tf.float32),
    )

    non_trainable_batch_norm_variables = []
    non_trainable_extra_variables = []
    for var in model.non_trainable_variables:
      if 'moving_mean' in var.name or 'moving_variance' in var.name:
        non_trainable_batch_norm_variables.append(var)
      else:
        non_trainable_extra_variables.append(var)

    logging.info(
        'Trainable model variables:\n%s',
        '\n'.join(
            [f'{var.name}\t{var.shape}' for var in model.trainable_variables]
        ),
    )
    logging.info(
        (
            'Non-trainable batch norm variables (get updated in training'
            ' mode):\n%s'
        ),
        '\n'.join(
            [
                f'{var.name}\t{var.shape}'
                for var in non_trainable_batch_norm_variables
            ]
        ),
    )
    logging.info(
        'Non-trainable frozen model variables:\n%s',
        '\n'.join(
            [
                f'{var.name}\t{var.shape}'
                for var in non_trainable_extra_variables
            ]
        ),
    )
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
        transform_and_batch_fn=batch_fn,
    )

    dataset = reader.read(input_context=input_context)

    return dataset

  def build_losses(
      self, labels, model_outputs, label_weights=None, aux_losses=None
  ):
    """Sigmoid Cross Entropy.

    Args:
      labels: tensor containing truth labels.
      model_outputs: output probabilities of the classifier.
      label_weights: optional tensor of label weights.
      aux_losses: tensor containing auxiliarly loss tensors, i.e. `losses` in
        keras.Model.

    Returns:
      A dict of tensors contains total loss, model loss tensors.
    """
    losses_config = self.task_config.losses
    model_loss = tf_keras.losses.binary_crossentropy(
        tf.expand_dims(labels, axis=-1),
        tf.expand_dims(model_outputs, axis=-1),
        from_logits=losses_config.from_logits,
        label_smoothing=losses_config.label_smoothing,
        axis=-1,
    )
    if label_weights is None:
      model_loss = tf_utils.safe_mean(model_loss)
    else:
      model_loss = model_loss * label_weights
      # Manutally compute weighted mean loss.
      total_loss = tf.reduce_sum(model_loss)
      total_weight = tf.cast(
          tf.reduce_sum(label_weights), dtype=total_loss.dtype
      )
      model_loss = tf.math.divide_no_nan(total_loss, total_weight)

    total_loss = model_loss
    if aux_losses:
      total_loss += tf.add_n(aux_losses)

    return {'total_loss': total_loss, 'model_loss': model_loss}

  def build_metrics(self, training=True):
    """Gets streaming metrics for training/validation.

       metric: mAP/gAP
       top_k: A positive integer specifying how many predictions are considered
        per video.
       top_n: A positive Integer specifying the average precision at n, or None
        to use all provided data points.
    Args:
      training: Bool value, true for training mode, false for eval/validation.

    Returns:
      A list of metrics to be used.
    """
    metrics = []
    metric_names = ['total_loss', 'model_loss']
    for name in metric_names:
      metrics.append(tf_keras.metrics.Mean(name, dtype=tf.float32))

    if (
        self.task_config.evaluation.average_precision is not None
        and not training
    ):
      # Cannot run in train step.
      num_classes = self.task_config.validation_data.num_classes
      top_k = self.task_config.evaluation.average_precision.top_k
      top_n = self.task_config.evaluation.average_precision.top_n
      self.avg_prec_metric = eval_util.EvaluationMetrics(
          num_classes, top_k=top_k, top_n=top_n
      )

    return metrics

  def process_metrics(
      self,
      metrics: List[tf_keras.metrics.Metric],
      labels: tf.Tensor,
      outputs: tf.Tensor,
      model_losses: Optional[Dict[str, tf.Tensor]] = None,
      label_weights: Optional[tf.Tensor] = None,
      training: bool = True,
      **kwargs,
  ) -> Dict[str, Tuple[tf.Tensor, ...]]:
    """Updates metrics.

    Args:
      metrics: Evaluation metrics to be updated.
      labels: A tensor containing truth labels.
      outputs: Model output logits of the classifier.
      model_losses: An optional dict of model losses.
      label_weights: Optional label weights, can be broadcast into shape of
        outputs/labels.
      training: Bool indicates if in training mode.
      **kwargs: Additional input arguments.

    Returns:
      Updated dict of metrics log.
    """
    if model_losses is None:
      model_losses = {}

    logs = {}
    if (
        self.task_config.evaluation.average_precision is not None
        and not training
    ):
      logs.update({self.avg_prec_metric.name: (labels, outputs)})

    for m in metrics:
      if m.name in model_losses:
        m.update_state(model_losses[m.name])
        logs[m.name] = m.result()
    return logs

  def _preprocess_model_inputs(
      self,
      inputs: dict[str, tf.Tensor],
      require_num_frames: bool = True,
      training: bool = True,
  ):
    """Preprocesses input tensors before model on device."""
    extra_inputs = {
        'num_frames': (
            tf.reshape(inputs['num_frames'], [-1])
            if require_num_frames
            else None
        ),
        'training': training,
    }
    return inputs['video_matrix'], extra_inputs

  def _preprocess_labels(
      self, inputs: dict[str, tf.Tensor], training: bool = True
  ):
    """Preprocesses labels."""
    del training  # training is unused in _preprocess_labels in YT8M.
    labels = inputs['labels']
    label_weights = inputs.get('label_weights', None)

    return labels, label_weights

  def _postprocess_outputs(
      self, outputs, labels, label_weights, training: bool = True
  ):
    """Postprocess model outputs (inputs / labels / label_weights)."""
    if not training and self.task_config.validation_data.segment_labels:
      # workaround to ignore the unrated labels.
      outputs *= label_weights
      # remove padding
      outputs = outputs[~tf.reduce_all(labels == -1, axis=1)]
      labels = labels[~tf.reduce_all(labels == -1, axis=1)]
    return outputs, labels, label_weights

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors. output_dict = { "video_ids":
        batch_video_ids, "video_matrix": batch_video_matrix, "labels":
        batch_labels, "num_frames": batch_frames, }
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      a dictionary of logs.
    """
    # Will require `num_frames` if `num_sample_frames` is None since
    # video_matrix is padded to max_frames in this case.
    require_num_frames = self.task_config.train_data.num_sample_frames is None
    inputs_tensor, extra_inputs = self._preprocess_model_inputs(
        inputs,
        require_num_frames=require_num_frames,
        training=True,
    )
    labels, label_weights = self._preprocess_labels(inputs, training=True)

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    with tf.GradientTape() as tape:
      outputs = model(inputs_tensor, **extra_inputs)['predictions']
      # Casting output layer as float32 is necessary when mixed_precision is
      # mixed_float16 or mixed_bfloat16 to ensure output is casted as float32.
      outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
      # Post-process model / label outputs.
      outputs, labels, label_weights = self._postprocess_outputs(
          outputs, labels, label_weights, training=True
      )

      # Computes per-replica loss
      all_losses = self.build_losses(
          model_outputs=outputs,
          labels=labels,
          label_weights=label_weights,
          aux_losses=model.losses,
      )

      loss = all_losses['total_loss']
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / num_replicas

      # For mixed_precision policy, when LossScaleOptimizer is used, loss is
      # scaled for numerical stability.
      if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    # Scales back gradient before apply_gradients when LossScaleOptimizer is
    # used.
    if isinstance(optimizer, tf_keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)

    # Apply gradient clipping.
    if self.task_config.gradient_clip_norm > 0:
      grads, _ = tf.clip_by_global_norm(
          grads, self.task_config.gradient_clip_norm
      )
    optimizer.apply_gradients(list(zip(grads, tvars)))

    logs = {self.loss: loss}
    logs.update(
        self.process_metrics(
            metrics,
            labels=labels,
            outputs=outputs,
            model_losses=all_losses,
            label_weights=label_weights,
            training=True,
        )
    )
    return logs

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors. output_dict = { "video_ids":
        batch_video_ids, "video_matrix": batch_video_matrix, "labels":
        batch_labels, "num_frames": batch_frames}.
      model: the model, forward definition.
      metrics: a nested structure of metrics objects.

    Returns:
      a dictionary of logs.
    """
    # Will require `num_frames` if `num_sample_frames` is None since
    # video_matrix is padded to max_frames in this case.
    require_num_frames = (
        self.task_config.validation_data.num_sample_frames is None
    )
    outputs = self.inference_step(
        model, inputs, require_num_frames=require_num_frames
    )['predictions']
    outputs = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), outputs)
    labels, label_weights = self._preprocess_labels(inputs, training=False)
    outputs, labels, label_weights = self._postprocess_outputs(
        outputs, labels, label_weights, training=False
    )

    all_losses = self.build_losses(
        labels=labels,
        model_outputs=outputs,
        label_weights=label_weights,
        aux_losses=model.losses,
    )

    logs = {self.loss: all_losses['total_loss']}
    logs.update(
        self.process_metrics(
            metrics,
            labels=labels,
            outputs=outputs,
            model_losses=all_losses,
            label_weights=inputs.get('label_weights', None),
            training=False,
        )
    )

    return logs

  def inference_step(self, model, inputs, require_num_frames=True):
    """Performs the forward step."""
    model_inputs, extra_inputs = self._preprocess_model_inputs(
        inputs, require_num_frames=require_num_frames, training=False
    )
    return model(model_inputs, **extra_inputs)

  def aggregate_logs(self, state=None, step_logs=None):
    if self.task_config.evaluation.average_precision is not None:
      if state is None:
        state = self.avg_prec_metric
      self.avg_prec_metric.accumulate(
          labels=step_logs[self.avg_prec_metric.name][0],
          predictions=step_logs[self.avg_prec_metric.name][1],
      )
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    if self.task_config.evaluation.average_precision is not None:
      avg_prec_metrics = self.avg_prec_metric.get(
          self.task_config.evaluation.average_precision.return_per_class_ap
      )
      self.avg_prec_metric.clear()
      return avg_prec_metrics
    return None
