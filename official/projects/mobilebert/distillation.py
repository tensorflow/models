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

"""Progressive distillation for MobileBERT student model."""
import dataclasses
from typing import List, Optional

from absl import logging
import orbit
import tensorflow as tf
from official.core import base_task
from official.core import config_definitions as cfg
from official.modeling import optimization
from official.modeling import tf_utils
from official.modeling.fast_training.progressive import policies
from official.modeling.hyperparams import base_config
from official.nlp import modeling
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.modeling import layers
from official.nlp.modeling import models


@dataclasses.dataclass
class LayerWiseDistillConfig(base_config.Config):
  """Defines the behavior of layerwise distillation."""
  num_steps: int = 10000
  warmup_steps: int = 0
  initial_learning_rate: float = 1.5e-3
  end_learning_rate: float = 1.5e-3
  decay_steps: int = 10000
  hidden_distill_factor: float = 100.0
  beta_distill_factor: float = 5000.0
  gamma_distill_factor: float = 5.0
  if_transfer_attention: bool = True
  attention_distill_factor: float = 1.0
  if_freeze_previous_layers: bool = False

  # The ids of teacher layers that will be mapped to the student model.
  # For example, if you want to compress a 24 layer teacher to a 6 layer
  # student, you can set it to [3, 7, 11, 15, 19, 23] (the index starts from 0).
  # If `None`, we assume teacher and student have the same number of layers,
  # and each layer of teacher model will be mapped to student's corresponding
  # layer.
  transfer_teacher_layers: Optional[List[int]] = None


@dataclasses.dataclass
class PretrainDistillConfig(base_config.Config):
  """Defines the behavior of pretrain distillation."""
  num_steps: int = 500000
  warmup_steps: int = 10000
  initial_learning_rate: float = 1.5e-3
  end_learning_rate: float = 1.5e-7
  decay_steps: int = 500000
  if_use_nsp_loss: bool = True
  distill_ground_truth_ratio: float = 0.5


@dataclasses.dataclass
class BertDistillationProgressiveConfig(policies.ProgressiveConfig):
  """Defines the specific distillation behavior."""
  if_copy_embeddings: bool = True
  layer_wise_distill_config: LayerWiseDistillConfig = dataclasses.field(
      default_factory=LayerWiseDistillConfig
  )
  pretrain_distill_config: PretrainDistillConfig = dataclasses.field(
      default_factory=PretrainDistillConfig
  )


@dataclasses.dataclass
class BertDistillationTaskConfig(cfg.TaskConfig):
  """Defines the teacher/student model architecture and training data."""
  teacher_model: bert.PretrainerConfig = dataclasses.field(
      default_factory=lambda: bert.PretrainerConfig(  # pylint: disable=g-long-lambda
          encoder=encoders.EncoderConfig(type='mobilebert')
      )
  )

  student_model: bert.PretrainerConfig = dataclasses.field(
      default_factory=lambda: bert.PretrainerConfig(  # pylint: disable=g-long-lambda
          encoder=encoders.EncoderConfig(type='mobilebert')
      )
  )
  # The path to the teacher model checkpoint or its directory.
  teacher_model_init_checkpoint: str = ''
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(
      default_factory=cfg.DataConfig
  )


def build_sub_encoder(encoder, target_layer_id):
  """Builds an encoder that only computes first few transformer layers."""
  input_ids = encoder.inputs[0]
  input_mask = encoder.inputs[1]
  type_ids = encoder.inputs[2]
  attention_mask = modeling.layers.SelfAttentionMask()(
      inputs=input_ids, to_mask=input_mask)
  embedding_output = encoder.embedding_layer(input_ids, type_ids)

  layer_output = embedding_output
  attention_score = None
  for layer_idx in range(target_layer_id + 1):
    layer_output, attention_score = encoder.transformer_layers[layer_idx](
        layer_output, attention_mask, return_attention_scores=True)

  return tf.keras.Model(
      inputs=[input_ids, input_mask, type_ids],
      outputs=[layer_output, attention_score])


class BertDistillationTask(policies.ProgressivePolicy, base_task.Task):
  """Distillation language modeling task progressively."""

  def __init__(self,
               strategy,
               progressive: BertDistillationProgressiveConfig,
               optimizer_config: optimization.OptimizationConfig,
               task_config: BertDistillationTaskConfig,
               logging_dir=None):

    self._strategy = strategy
    self._task_config = task_config
    self._progressive_config = progressive
    self._optimizer_config = optimizer_config
    self._train_data_config = task_config.train_data
    self._eval_data_config = task_config.validation_data
    self._the_only_train_dataset = None
    self._the_only_eval_dataset = None

    layer_wise_config = self._progressive_config.layer_wise_distill_config
    transfer_teacher_layers = layer_wise_config.transfer_teacher_layers
    num_teacher_layers = (
        self._task_config.teacher_model.encoder.mobilebert.num_blocks)
    num_student_layers = (
        self._task_config.student_model.encoder.mobilebert.num_blocks)
    if transfer_teacher_layers and len(
        transfer_teacher_layers) != num_student_layers:
      raise ValueError('The number of `transfer_teacher_layers` %s does not '
                       'match the number of student layers. %d' %
                       (transfer_teacher_layers, num_student_layers))
    if not transfer_teacher_layers and (num_teacher_layers !=
                                        num_student_layers):
      raise ValueError('`transfer_teacher_layers` is not specified, and the '
                       'number of teacher layers does not match '
                       'the number of student layers.')

    ratio = progressive.pretrain_distill_config.distill_ground_truth_ratio
    if ratio < 0 or ratio > 1:
      raise ValueError('distill_ground_truth_ratio has to be within [0, 1].')

    # A non-trainable layer for feature normalization for transfer loss
    self._layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        beta_initializer='zeros',
        gamma_initializer='ones',
        trainable=False)

    # Build the teacher and student pretrainer model.
    self._teacher_pretrainer = self._build_pretrainer(
        self._task_config.teacher_model, name='teacher')
    self._student_pretrainer = self._build_pretrainer(
        self._task_config.student_model, name='student')

    base_task.Task.__init__(
        self, params=task_config, logging_dir=logging_dir)
    policies.ProgressivePolicy.__init__(self)

  def _build_pretrainer(self, pretrainer_cfg: bert.PretrainerConfig, name: str):
    """Builds pretrainer from config and encoder."""
    encoder = encoders.build_encoder(pretrainer_cfg.encoder)
    if pretrainer_cfg.cls_heads:
      cls_heads = [
          layers.ClassificationHead(**cfg.as_dict())
          for cfg in pretrainer_cfg.cls_heads
      ]
    else:
      cls_heads = []

    masked_lm = layers.MobileBertMaskedLM(
        embedding_table=encoder.get_embedding_table(),
        activation=tf_utils.get_activation(pretrainer_cfg.mlm_activation),
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=pretrainer_cfg.mlm_initializer_range),
        name='cls/predictions')

    pretrainer = models.BertPretrainerV2(
        encoder_network=encoder,
        classification_heads=cls_heads,
        customized_masked_lm=masked_lm,
        name=name)
    return pretrainer

  # override policies.ProgressivePolicy
  def num_stages(self):
    # One stage for each layer, plus additional stage for pre-training
    return self._task_config.student_model.encoder.mobilebert.num_blocks + 1

  # override policies.ProgressivePolicy
  def num_steps(self, stage_id) -> int:
    """Return the total number of steps in this stage."""
    if stage_id + 1 < self.num_stages():
      return self._progressive_config.layer_wise_distill_config.num_steps
    else:
      return self._progressive_config.pretrain_distill_config.num_steps

  # override policies.ProgressivePolicy
  def get_model(self, stage_id, old_model=None) -> tf.keras.Model:
    del old_model
    return self.build_model(stage_id)

  # override policies.ProgressivePolicy
  def get_optimizer(self, stage_id):
    """Build optimizer for each stage."""
    if stage_id + 1 < self.num_stages():
      distill_config = self._progressive_config.layer_wise_distill_config
    else:
      distill_config = self._progressive_config.pretrain_distill_config

    params = self._optimizer_config.replace(
        learning_rate={
            'polynomial': {
                'decay_steps':
                    distill_config.decay_steps,
                'initial_learning_rate':
                    distill_config.initial_learning_rate,
                'end_learning_rate':
                    distill_config.end_learning_rate,
            }
        },
        warmup={
            'linear':
                {'warmup_steps':
                     distill_config.warmup_steps,
                }
            })
    opt_factory = optimization.OptimizerFactory(params)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    if isinstance(optimizer, tf.keras.optimizers.experimental.Optimizer):
      optimizer = tf.keras.__internal__.optimizers.convert_to_legacy_optimizer(
          optimizer)

    return optimizer

  # override policies.ProgressivePolicy
  def get_train_dataset(self, stage_id: int) -> tf.data.Dataset:
    """Return Dataset for this stage."""
    del stage_id
    if self._the_only_train_dataset is None:
      self._the_only_train_dataset = orbit.utils.make_distributed_dataset(
          self._strategy, self.build_inputs, self._train_data_config)
    return self._the_only_train_dataset

  # overrides policies.ProgressivePolicy
  def get_eval_dataset(self, stage_id):
    del stage_id
    if self._the_only_eval_dataset is None:
      self._the_only_eval_dataset = orbit.utils.make_distributed_dataset(
          self._strategy, self.build_inputs, self._eval_data_config)
    return self._the_only_eval_dataset

  # override base_task.task
  def build_model(self, stage_id) -> tf.keras.Model:
    """Build teacher/student keras models with outputs for current stage."""
    # Freeze the teacher model.
    self._teacher_pretrainer.trainable = False
    layer_wise_config = self._progressive_config.layer_wise_distill_config
    freeze_previous_layers = layer_wise_config.if_freeze_previous_layers
    student_encoder = self._student_pretrainer.encoder_network

    if stage_id != self.num_stages() - 1:
      # Build a model that outputs teacher's and student's transformer outputs.
      inputs = student_encoder.inputs
      student_sub_encoder = build_sub_encoder(
          encoder=student_encoder, target_layer_id=stage_id)
      student_output_feature, student_attention_score = student_sub_encoder(
          inputs)

      if layer_wise_config.transfer_teacher_layers:
        teacher_layer_id = layer_wise_config.transfer_teacher_layers[stage_id]
      else:
        teacher_layer_id = stage_id

      teacher_sub_encoder = build_sub_encoder(
          encoder=self._teacher_pretrainer.encoder_network,
          target_layer_id=teacher_layer_id)

      teacher_output_feature, teacher_attention_score = teacher_sub_encoder(
          inputs)

      if freeze_previous_layers:
        student_encoder.embedding_layer.trainable = False
        for i in range(stage_id):
          student_encoder.transformer_layers[i].trainable = False

      return tf.keras.Model(
          inputs=inputs,
          outputs=dict(
              student_output_feature=student_output_feature,
              student_attention_score=student_attention_score,
              teacher_output_feature=teacher_output_feature,
              teacher_attention_score=teacher_attention_score))
    else:
      # Build a model that outputs teacher's and student's MLM/NSP outputs.
      inputs = self._student_pretrainer.inputs
      student_pretrainer_output = self._student_pretrainer(inputs)
      teacher_pretrainer_output = self._teacher_pretrainer(inputs)

      # Set all student's transformer blocks to trainable.
      if freeze_previous_layers:
        student_encoder.embedding_layer.trainable = True
        for layer in student_encoder.transformer_layers:
          layer.trainable = True

      model = tf.keras.Model(
          inputs=inputs,
          outputs=dict(
              student_pretrainer_output=student_pretrainer_output,
              teacher_pretrainer_output=teacher_pretrainer_output,
          ))
      # Checkpoint the student encoder which is the goal of distillation.
      model.checkpoint_items = self._student_pretrainer.checkpoint_items
      return model

  # overrides base_task.Task
  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for pretraining."""
    # copy from masked_lm.py for testing
    if params.input_path == 'dummy':

      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        dummy_lm = tf.zeros((1, params.max_predictions_per_seq), dtype=tf.int32)
        return dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids,
            masked_lm_positions=dummy_lm,
            masked_lm_ids=dummy_lm,
            masked_lm_weights=tf.cast(dummy_lm, dtype=tf.float32),
            next_sentence_labels=tf.zeros((1, 1), dtype=tf.int32))

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return data_loader_factory.get_data_loader(params).load(input_context)

  def _get_distribution_losses(self, teacher, student):
    """Return the beta and gamma distall losses for feature distribution."""
    teacher_mean = tf.math.reduce_mean(teacher, axis=-1, keepdims=True)
    student_mean = tf.math.reduce_mean(student, axis=-1, keepdims=True)
    teacher_var = tf.math.reduce_variance(teacher, axis=-1, keepdims=True)
    student_var = tf.math.reduce_variance(student, axis=-1, keepdims=True)

    beta_loss = tf.math.squared_difference(student_mean, teacher_mean)
    beta_loss = tf.math.reduce_mean(beta_loss, axis=None, keepdims=False)
    gamma_loss = tf.math.abs(student_var - teacher_var)
    gamma_loss = tf.math.reduce_mean(gamma_loss, axis=None, keepdims=False)

    return beta_loss, gamma_loss

  def _get_attention_loss(self, teacher_score, student_score):
    # Note that the definition of KLDivergence here is a little different from
    # the original one (tf.keras.losses.KLDivergence). We adopt this approach
    # to stay consistent with the TF1 implementation.
    teacher_weight = tf.keras.activations.softmax(teacher_score, axis=-1)
    student_log_weight = tf.nn.log_softmax(student_score, axis=-1)
    kl_divergence = -(teacher_weight * student_log_weight)
    kl_divergence = tf.math.reduce_sum(kl_divergence, axis=-1, keepdims=True)
    kl_divergence = tf.math.reduce_mean(kl_divergence, axis=None,
                                        keepdims=False)
    return kl_divergence

  def build_losses(self, labels, outputs, metrics) -> tf.Tensor:
    """Builds losses and update loss-related metrics for the current stage."""
    last_stage = 'student_pretrainer_output' in outputs

    # Layer-wise warmup stage
    if not last_stage:
      distill_config = self._progressive_config.layer_wise_distill_config
      teacher_feature = outputs['teacher_output_feature']
      student_feature = outputs['student_output_feature']

      feature_transfer_loss = tf.keras.losses.mean_squared_error(
          self._layer_norm(teacher_feature), self._layer_norm(student_feature))
      feature_transfer_loss *= distill_config.hidden_distill_factor
      beta_loss, gamma_loss = self._get_distribution_losses(teacher_feature,
                                                            student_feature)
      beta_loss *= distill_config.beta_distill_factor
      gamma_loss *= distill_config.gamma_distill_factor
      total_loss = feature_transfer_loss + beta_loss + gamma_loss

      if distill_config.if_transfer_attention:
        teacher_attention = outputs['teacher_attention_score']
        student_attention = outputs['student_attention_score']
        attention_loss = self._get_attention_loss(teacher_attention,
                                                  student_attention)
        attention_loss *= distill_config.attention_distill_factor
        total_loss += attention_loss

      total_loss /= tf.cast((self._stage_id + 1), tf.float32)

    # Last stage to distill pretraining layer.
    else:
      distill_config = self._progressive_config.pretrain_distill_config
      lm_label = labels['masked_lm_ids']
      vocab_size = (
          self._task_config.student_model.encoder.mobilebert.word_vocab_size)

      # Shape: [batch, max_predictions_per_seq, vocab_size]
      lm_label = tf.one_hot(indices=lm_label, depth=vocab_size, on_value=1.0,
                            off_value=0.0, axis=-1, dtype=tf.float32)
      gt_ratio = distill_config.distill_ground_truth_ratio
      if gt_ratio != 1.0:
        teacher_mlm_logits = outputs['teacher_pretrainer_output']['mlm_logits']
        teacher_labels = tf.nn.softmax(teacher_mlm_logits, axis=-1)
        lm_label = gt_ratio * lm_label + (1-gt_ratio) * teacher_labels

      student_pretrainer_output = outputs['student_pretrainer_output']
      # Shape: [batch, max_predictions_per_seq, vocab_size]
      student_lm_log_probs = tf.nn.log_softmax(
          student_pretrainer_output['mlm_logits'], axis=-1)

      # Shape: [batch * max_predictions_per_seq]
      per_example_loss = tf.reshape(
          -tf.reduce_sum(student_lm_log_probs * lm_label, axis=[-1]), [-1])

      lm_label_weights = tf.reshape(labels['masked_lm_weights'], [-1])
      lm_numerator_loss = tf.reduce_sum(per_example_loss * lm_label_weights)
      lm_denominator_loss = tf.reduce_sum(lm_label_weights)
      mlm_loss = tf.math.divide_no_nan(lm_numerator_loss, lm_denominator_loss)
      total_loss = mlm_loss

      if 'next_sentence_labels' in labels:
        sentence_labels = labels['next_sentence_labels']
        sentence_outputs = tf.cast(
            student_pretrainer_output['next_sentence'], dtype=tf.float32)
        sentence_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                sentence_labels, sentence_outputs, from_logits=True))
        total_loss += sentence_loss

    # Also update loss-related metrics here, instead of in `process_metrics`.
    metrics = dict([(metric.name, metric) for metric in metrics])

    if not last_stage:
      metrics['feature_transfer_mse'].update_state(feature_transfer_loss)
      metrics['beta_transfer_loss'].update_state(beta_loss)
      metrics['gamma_transfer_loss'].update_state(gamma_loss)
      layer_wise_config = self._progressive_config.layer_wise_distill_config
      if layer_wise_config.if_transfer_attention:
        metrics['attention_transfer_loss'].update_state(attention_loss)
    else:
      metrics['lm_example_loss'].update_state(mlm_loss)
      if 'next_sentence_labels' in labels:
        metrics['next_sentence_loss'].update_state(sentence_loss)
    metrics['total_loss'].update_state(total_loss)

    return total_loss

  # overrides base_task.Task
  def build_metrics(self, training=None):
    del training
    metrics = [
        tf.keras.metrics.Mean(name='feature_transfer_mse'),
        tf.keras.metrics.Mean(name='beta_transfer_loss'),
        tf.keras.metrics.Mean(name='gamma_transfer_loss'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='masked_lm_accuracy'),
        tf.keras.metrics.Mean(name='lm_example_loss'),
        tf.keras.metrics.Mean(name='total_loss')]
    if self._progressive_config.layer_wise_distill_config.if_transfer_attention:
      metrics.append(tf.keras.metrics.Mean(name='attention_transfer_loss'))
    if self._task_config.train_data.use_next_sentence_label:
      metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(
          name='next_sentence_accuracy'))
      metrics.append(tf.keras.metrics.Mean(name='next_sentence_loss'))

    return metrics

  # overrides base_task.Task
  # process non-loss metrics
  def process_metrics(self, metrics, labels, student_pretrainer_output):
    metrics = dict([(metric.name, metric) for metric in metrics])
    # Final pretrainer layer distillation stage.
    if student_pretrainer_output is not None:
      if 'masked_lm_accuracy' in metrics:
        metrics['masked_lm_accuracy'].update_state(
            labels['masked_lm_ids'], student_pretrainer_output['mlm_logits'],
            labels['masked_lm_weights'])
      if 'next_sentence_accuracy' in metrics:
        metrics['next_sentence_accuracy'].update_state(
            labels['next_sentence_labels'],
            student_pretrainer_output['next_sentence'])

  # overrides base_task.Task
  def train_step(self, inputs, model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer, metrics):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    with tf.GradientTape() as tape:
      outputs = model(inputs, training=True)

      # Computes per-replica loss.
      loss = self.build_losses(
          labels=inputs,
          outputs=outputs,
          metrics=metrics)
    # Scales loss as the default gradients allreduce performs sum inside the
    # optimizer.
    # TODO(b/154564893): enable loss scaling.
    # scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync

    # get trainable variables for current stage
    tvars = model.trainable_variables
    last_stage = 'student_pretrainer_output' in outputs

    grads = tape.gradient(loss, tvars)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    self.process_metrics(
        metrics, inputs,
        outputs['student_pretrainer_output'] if last_stage else None)
    return {self.loss: loss}

  # overrides base_task.Task
  def validation_step(self, inputs, model: tf.keras.Model, metrics):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    outputs = model(inputs, training=False)
    # Computes per-replica loss.
    loss = self.build_losses(labels=inputs, outputs=outputs, metrics=metrics)
    last_stage = 'student_pretrainer_output' in outputs
    self.process_metrics(
        metrics, inputs,
        outputs['student_pretrainer_output'] if last_stage else None)
    return {self.loss: loss}

  @property
  def cur_checkpoint_items(self):
    """Checkpoints for model, stage_id, optimizer for preemption handling."""
    return dict(
        stage_id=self._stage_id,
        volatiles=self._volatiles,
        student_pretrainer=self._student_pretrainer,
        teacher_pretrainer=self._teacher_pretrainer,
        encoder=self._student_pretrainer.encoder_network)

  def initialize(self, model):
    """Loads teacher's pretrained checkpoint and copy student's embedding."""
    # This function will be called when no checkpoint found for the model,
    # i.e., when the training starts (not preemption case).
    # The weights of teacher pretrainer and student pretrainer will be
    # initialized, rather than the passed-in `model`.
    del model
    logging.info('Begin to load checkpoint for teacher pretrainer model.')
    ckpt_dir_or_file = self._task_config.teacher_model_init_checkpoint
    if not ckpt_dir_or_file:
      raise ValueError('`teacher_model_init_checkpoint` is not specified.')

    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    # Makes sure the teacher pretrainer variables are created.
    _ = self._teacher_pretrainer(self._teacher_pretrainer.inputs)
    teacher_checkpoint = tf.train.Checkpoint(
        **self._teacher_pretrainer.checkpoint_items)
    teacher_checkpoint.read(ckpt_dir_or_file).assert_existing_objects_matched()

    logging.info('Begin to copy word embedding from teacher model to student.')
    teacher_encoder = self._teacher_pretrainer.encoder_network
    student_encoder = self._student_pretrainer.encoder_network
    embedding_weights = teacher_encoder.embedding_layer.get_weights()
    student_encoder.embedding_layer.set_weights(embedding_weights)
