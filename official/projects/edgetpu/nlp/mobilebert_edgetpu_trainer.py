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

"""Distillation trainer for EdgeTPU-BERT."""
import enum
import os
from typing import Optional

from absl import logging
import orbit
import tensorflow as tf, tf_keras

from official.modeling import optimization
from official.nlp import modeling
from official.nlp.data import data_loader_factory
from official.projects.edgetpu.nlp.configs import params


class DistillationMode(enum.Enum):
  """enum.Enum class for different distillation mode.

  A state machine is used to control the training progress. When the training
  job starts from the beginning or resumes from a preemption, the state is INIT.
  Then depends on the 'self.current_step', the state switches to either
  'LAYER_WISE' or 'END2END'.

  Options:
    UNKNOWN: Unknown status, always raise errors.
    INIT: The trainer is initialized or restarted from the preemption.
    LAYER_WISE: Layer-wise distillation for each transformer layers.
    END2END: End-to-end distillation after layer-wise distillaiton is done.
  """
  UNKNOWN = 0
  INIT = 1
  LAYER_WISE = 2
  END2END = 3


def _get_distribution_losses(teacher, student):
  """Returns the beta and gamma distall losses for feature distribution."""
  teacher_mean = tf.math.reduce_mean(teacher, axis=-1, keepdims=True)
  student_mean = tf.math.reduce_mean(student, axis=-1, keepdims=True)
  teacher_var = tf.math.reduce_variance(teacher, axis=-1, keepdims=True)
  student_var = tf.math.reduce_variance(student, axis=-1, keepdims=True)

  beta_loss = tf.math.squared_difference(student_mean, teacher_mean)
  beta_loss = tf.math.reduce_mean(beta_loss, axis=None, keepdims=False)
  gamma_loss = tf.math.abs(student_var - teacher_var)
  gamma_loss = tf.math.reduce_mean(gamma_loss, axis=None, keepdims=False)

  return beta_loss, gamma_loss


def _get_attention_loss(teacher_score, student_score):
  """Function to calculate attention loss for transformer layers."""
  # Note that the definition of KLDivergence here is a little different from
  # the original one (tf_keras.losses.KLDivergence). We adopt this approach
  # to stay consistent with the TF1 implementation.
  teacher_weight = tf_keras.activations.softmax(teacher_score, axis=-1)
  student_log_weight = tf.nn.log_softmax(student_score, axis=-1)
  kl_divergence = -(teacher_weight * student_log_weight)
  kl_divergence = tf.math.reduce_sum(kl_divergence, axis=-1, keepdims=True)
  kl_divergence = tf.math.reduce_mean(kl_divergence, axis=None,
                                      keepdims=False)
  return kl_divergence


def _build_sub_encoder(encoder, stage_number):
  """Builds a partial model containing the first few transformer layers."""
  input_ids = encoder.inputs[0]
  input_mask = encoder.inputs[1]
  type_ids = encoder.inputs[2]
  attention_mask = modeling.layers.SelfAttentionMask()(
      inputs=input_ids, to_mask=input_mask)
  embedding_output = encoder.embedding_layer(input_ids, type_ids)

  layer_output = embedding_output
  attention_score = None
  for layer_idx in range(stage_number + 1):
    layer_output, attention_score = encoder.transformer_layers[layer_idx](
        layer_output, attention_mask, return_attention_scores=True)

  return tf_keras.Model(
      inputs=[input_ids, input_mask, type_ids],
      outputs=[layer_output, attention_score])


class MobileBERTEdgeTPUDistillationTrainer(orbit.StandardTrainer,
                                           orbit.StandardEvaluator):
  """Orbit based distillation training pipeline for MobileBERT-EdgeTPU models."""

  def __init__(self,
               teacher_model: modeling.models.BertPretrainerV2,
               student_model: modeling.models.BertPretrainerV2,
               strategy: tf.distribute.Strategy,
               experiment_params: params.EdgeTPUBERTCustomParams,
               export_ckpt_path: Optional[str] = None,
               reuse_teacher_embedding: Optional[bool] = True):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.strategy = strategy
    self.layer_wise_distill_config = experiment_params.layer_wise_distillation
    self.e2e_distill_config = experiment_params.end_to_end_distillation
    self.optimizer_config = experiment_params.optimizer
    self.train_dataset_config = experiment_params.train_datasest
    self.eval_dataset_config = experiment_params.eval_dataset
    self.word_vocab_size = experiment_params.student_model.encoder.mobilebert.word_vocab_size
    self.distill_gt_ratio = experiment_params.end_to_end_distillation.distill_ground_truth_ratio
    self.teacher_transformer_layers = experiment_params.teacher_model.encoder.mobilebert.num_blocks
    self.student_transformer_layers = experiment_params.student_model.encoder.mobilebert.num_blocks
    self.exported_ckpt_path = export_ckpt_path
    self.current_step = orbit.utils.create_global_step()
    self.current_step.assign(0)

    # Stage is updated every time when the distillation is done for one
    # transformer layer. self.stage is updated at the train_loop_begin()
    # function. After the last stage is done, the self.mode is changed to
    # 'e2e'.
    self.stage = 0
    self.mode = DistillationMode.INIT

    # Number of transformer layers in teacher should be equal (or divisible)
    # by the number of transformer layers in student.
    if self.teacher_transformer_layers % self.student_transformer_layers != 0:
      raise ValueError(
          'Number of transformer layer must be equal or divisible.')
    self.ratio = (self.teacher_transformer_layers //
                  self.student_transformer_layers)

    # Create optimizers for different training stage.
    self.layer_wise_optimizer = self.build_optimizer(
        self.layer_wise_distill_config)
    self.e2e_optimizer = self.build_optimizer(self.e2e_distill_config)
    self.current_optimizer = self.layer_wise_optimizer

    # A non-trainable layer for feature normalization for transfer loss.
    self._layer_norm = tf_keras.layers.LayerNormalization(
        axis=-1,
        beta_initializer='zeros',
        gamma_initializer='ones',
        trainable=False)

    self.build_dataset()
    self.build_metrics()

    # Create an empty exported checkpoint manager, it will be initialized once
    # the training mode enters END2END.
    self.exported_ckpt_manager = None

    # Reuse the teacher's embedding table in student model.
    if reuse_teacher_embedding:
      logging.info('Copy word embedding from teacher model to student.')
      teacher_encoder = self.teacher_model.encoder_network
      student_encoder = self.student_model.encoder_network
      embedding_weights = teacher_encoder.embedding_layer.get_weights()
      student_encoder.embedding_layer.set_weights(embedding_weights)

    orbit.StandardTrainer.__init__(self, self.train_dataset)
    orbit.StandardEvaluator.__init__(self, self.eval_dataset)

  def build_dataset(self):
    """Creates the training and evaluation dataset."""
    # Returns None when the input_path is 'dummy'.
    if self.train_dataset_config.input_path == 'dummy':
      self.train_dataset = None
      self.eval_dataset = None
      return
    # None distributed dataset.
    train_dataset = data_loader_factory.get_data_loader(
        self.train_dataset_config).load()
    eval_dataset = data_loader_factory.get_data_loader(
        self.eval_dataset_config).load()
    # Ddistributed dataset.
    self.train_dataset = orbit.utils.make_distributed_dataset(
        self.strategy, train_dataset)
    self.eval_dataset = orbit.utils.make_distributed_dataset(
        self.strategy, eval_dataset)

  def build_model(self):
    """Creates the fused model from teacher/student model."""
    self.teacher_model.trainable = False

    if self.mode == DistillationMode.LAYER_WISE:
      # Build a model that outputs teacher's and student's transformer outputs.
      inputs = self.student_model.encoder_network.inputs
      student_sub_encoder = _build_sub_encoder(
          encoder=self.student_model.encoder_network,
          stage_number=self.stage)
      student_output_feature, student_attention_score = student_sub_encoder(
          inputs)
      teacher_sub_encoder = _build_sub_encoder(
          encoder=self.teacher_model.encoder_network,
          stage_number=int(self.stage * self.ratio))
      teacher_output_feature, teacher_attention_score = teacher_sub_encoder(
          inputs)
      return tf_keras.Model(
          inputs=inputs,
          outputs=dict(
              student_output_feature=student_output_feature,
              student_attention_score=student_attention_score,
              teacher_output_feature=teacher_output_feature,
              teacher_attention_score=teacher_attention_score))
    elif self.mode == DistillationMode.END2END:
      # Build a model that outputs teacher's and student's MLM/NSP outputs.
      inputs = self.student_model.inputs
      student_pretrainer_outputs = self.student_model(inputs)
      teacher_pretrainer_outputs = self.teacher_model(inputs)
      model = tf_keras.Model(
          inputs=inputs,
          outputs=dict(
              student_pretrainer_outputs=student_pretrainer_outputs,
              teacher_pretrainer_outputs=teacher_pretrainer_outputs,
          ))
      # Checkpoint the student encoder which is the goal of distillation.
      model.checkpoint_items = self.student_model.checkpoint_items
      return model
    else:
      raise ValueError(f'Unknown distillation mode: {self.mode}.')

  def build_optimizer(self, config):
    """Creates optimier for the fused model."""
    optimizer_config = self.optimizer_config.replace(
        learning_rate={
            'polynomial': {
                'decay_steps': config.decay_steps,
                'initial_learning_rate': config.initial_learning_rate,
                'end_learning_rate': config.end_learning_rate,
            }
        },
        warmup={
            'type': 'linear',
            'linear': {
                'warmup_steps': config.warmup_steps,
            }
        })
    logging.info('The optimizer config is: %s', optimizer_config.as_dict())

    optimizer_factory = optimization.OptimizerFactory(optimizer_config)
    return optimizer_factory.build_optimizer(
        optimizer_factory.build_learning_rate())

  def build_metrics(self):
    """Creates metrics functions for the training."""
    self.train_metrics = {
        'feature_transfer_mse': tf_keras.metrics.Mean(),
        'beta_transfer_loss': tf_keras.metrics.Mean(),
        'gamma_transfer_loss': tf_keras.metrics.Mean(),
        'attention_transfer_loss': tf_keras.metrics.Mean(),
        'masked_lm_accuracy': tf_keras.metrics.SparseCategoricalAccuracy(),
        'lm_example_loss': tf_keras.metrics.Mean(),
        'total_loss': tf_keras.metrics.Mean(),
        'next_sentence_accuracy': tf_keras.metrics.SparseCategoricalAccuracy(),
        'next_sentence_loss': tf_keras.metrics.Mean(),
    }
    self.eval_metrics = {
        'masked_lm_accuracy': tf_keras.metrics.SparseCategoricalAccuracy(),
        'next_sentence_accuracy': tf_keras.metrics.SparseCategoricalAccuracy(),
    }

  def build_exported_ckpt_manager(self):
    """Creates checkpoint manager for exported models."""
    if self.exported_ckpt_path is None:
      logging.warn('exported_ckpt_path is not specified. The saved model'
                   'can not be used for downstreaming tasks.')
      return
    checkpoint = tf.train.Checkpoint(global_step=self.current_step,
                                     model=self.model,
                                     optimizer=self.current_optimizer,
                                     **self.model.checkpoint_items)
    self.exported_ckpt_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=os.path.join(self.exported_ckpt_path, 'exported_ckpt'),
        max_to_keep=2,
        step_counter=self.current_step,
        checkpoint_interval=20000,
        init_fn=None)

  def calculate_loss_metrics(self, labels, outputs):
    """Calculates loss and metrics.

    Args:
      labels: Ground truth from dataset.
      outputs: fused outputs from teacher model and student model.
    Returns:
      total loss value.
    """
    if self.mode == DistillationMode.LAYER_WISE:
      teacher_feature = outputs['teacher_output_feature']
      student_feature = outputs['student_output_feature']
      feature_transfer_loss = tf_keras.losses.mean_squared_error(
          self._layer_norm(teacher_feature), self._layer_norm(student_feature))
      # feature_transfer_loss = tf.reduce_mean(feature_transfer_loss)
      feature_transfer_loss *= self.layer_wise_distill_config.hidden_distill_factor
      beta_loss, gamma_loss = _get_distribution_losses(teacher_feature,
                                                       student_feature)
      beta_loss *= self.layer_wise_distill_config.beta_distill_factor
      gamma_loss *= self.layer_wise_distill_config.gamma_distill_factor
      total_loss = feature_transfer_loss + beta_loss + gamma_loss

      teacher_attention = outputs['teacher_attention_score']
      student_attention = outputs['student_attention_score']
      attention_loss = _get_attention_loss(teacher_attention, student_attention)
      attention_loss *= self.layer_wise_distill_config.attention_distill_factor
      total_loss += attention_loss
      total_loss /= tf.cast((self.stage + 1), tf.float32)
    elif self.mode == DistillationMode.END2END:
      lm_label = labels['masked_lm_ids']
      # Shape: [batch, max_predictions_per_seq, word_vocab_size]
      lm_label = tf.one_hot(indices=lm_label,
                            depth=self.word_vocab_size,
                            on_value=1.0,
                            off_value=0.0,
                            axis=-1,
                            dtype=tf.float32)
      lm_label_weights = labels['masked_lm_weights']
      teacher_mlm_logits = outputs['teacher_pretrainer_outputs']['mlm_logits']
      teacher_labels = tf.nn.softmax(teacher_mlm_logits, axis=-1)
      gt_label = self.distill_gt_ratio * lm_label
      teacher_label = (1 - self.distill_gt_ratio) * teacher_labels
      lm_label = gt_label + teacher_label

      student_pretrainer_output = outputs['student_pretrainer_outputs']
      # Shape: [batch, max_predictions_per_seq, word_vocab_size]
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

      sentence_labels = labels['next_sentence_labels']
      sentence_outputs = tf.cast(
          student_pretrainer_output['next_sentence'], dtype=tf.float32)
      sentence_loss = tf.reduce_mean(
          tf_keras.losses.sparse_categorical_crossentropy(
              sentence_labels, sentence_outputs, from_logits=True))
      total_loss += sentence_loss
    else:
      raise ValueError('Training mode has to be LAYER-WISE or END2END.')

    if self.mode == DistillationMode.LAYER_WISE:
      self.train_metrics['feature_transfer_mse'].update_state(
          feature_transfer_loss)
      self.train_metrics['beta_transfer_loss'].update_state(beta_loss)
      self.train_metrics['gamma_transfer_loss'].update_state(gamma_loss)
      self.train_metrics['attention_transfer_loss'].update_state(attention_loss)
    elif self.mode == DistillationMode.END2END:
      self.train_metrics['lm_example_loss'].update_state(mlm_loss)
      self.train_metrics['next_sentence_loss'].update_state(sentence_loss)
    self.train_metrics['total_loss'].update_state(total_loss)

    return total_loss

  def calculate_accuracy_metrics(self, labels, outputs, metrics):
    """Calculates metrics that are not related to the losses."""
    if self.mode == DistillationMode.END2END:
      student_pretrainer_output = outputs['student_pretrainer_outputs']
      metrics['masked_lm_accuracy'].update_state(
          labels['masked_lm_ids'],
          student_pretrainer_output['mlm_logits'],
          labels['masked_lm_weights'])
      metrics['next_sentence_accuracy'].update_state(
          labels['next_sentence_labels'],
          student_pretrainer_output['next_sentence'])

  def _rebuild_training_graph(self):
    """Rebuilds the training graph when one stage/step is done."""
    self.stage = (self.current_step.numpy() //
                  self.layer_wise_distill_config.num_steps)
    logging.info('Start distillation training for the %d stage', self.stage)
    self.model = self.build_model()
    self.layer_wise_optimizer = self.build_optimizer(
        self.layer_wise_distill_config)
    # Rebuild the dataset which can significantly improve the training
    # accuracy.
    logging.info('Rebuild the training dataset.')
    self.build_dataset()
    # Setting `self._train_loop_fn` and `self._eval_loop_fn` to None will
    # rebuild the train and eval functions with the updated loss function.
    logging.info('Rebuild the training and evaluation graph.')
    self._train_loop_fn = None
    self._eval_loop_fn = None

  def train_loop_begin(self):
    """A train loop is similar with the concept of an epoch."""
    self.train_metrics['feature_transfer_mse'].reset_states()
    self.train_metrics['beta_transfer_loss'].reset_states()
    self.train_metrics['gamma_transfer_loss'].reset_states()
    self.train_metrics['attention_transfer_loss'].reset_states()
    self.train_metrics['total_loss'].reset_states()
    self.train_metrics['lm_example_loss'].reset_states()
    self.train_metrics['next_sentence_loss'].reset_states()
    self.train_metrics['masked_lm_accuracy'].reset_states()
    self.train_metrics['next_sentence_accuracy'].reset_states()

    if self.mode == DistillationMode.INIT:
      if (self.current_step.numpy() < self.layer_wise_distill_config.num_steps *
          self.student_transformer_layers):
        logging.info('Start or resume layer-wise training.')
        self.mode = DistillationMode.LAYER_WISE
        self.stage = (self.current_step.numpy() //
                      self.layer_wise_distill_config.num_steps)
        self.model = self.build_model()
        self.build_dataset()
        self.current_optimizer = self.layer_wise_optimizer
      else:
        self.mode = DistillationMode.END2END
        logging.info('Start or resume e2e training.')
        self.model = self.build_model()
        self.current_optimizer = self.e2e_optimizer
    elif self.mode == DistillationMode.LAYER_WISE:
      if (self.current_step.numpy() < self.layer_wise_distill_config.num_steps *
          self.student_transformer_layers):
        if (self.current_step.numpy() %
            self.layer_wise_distill_config.num_steps) == 0:
          self._rebuild_training_graph()
        self.current_optimizer = self.layer_wise_optimizer
      else:
        self.mode = DistillationMode.END2END
        self.model = self.build_model()
        logging.info('Start e2e distillation training.')
        self.current_optimizer = self.e2e_optimizer
        logging.info('Rebuild the training dataset.')
        self.build_dataset()
        logging.info('Rebuild the training and evaluation graph.')
        self._train_loop_fn = None
        self._eval_loop_fn = None

  def train_step(self, iterator):
    """A single step of train."""

    def step_fn(inputs):
      with tf.GradientTape() as tape:
        outputs = self.model(inputs, training=True)
        loss = self.calculate_loss_metrics(inputs, outputs)
        self.calculate_accuracy_metrics(inputs, outputs, self.train_metrics)

      grads = tape.gradient(loss, self.model.trainable_variables)
      self.current_optimizer.apply_gradients(
          zip(grads, self.model.trainable_variables))
      self.current_step.assign_add(1)

    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """A train loop is similar with the concept of an epoch."""
    if self.mode == DistillationMode.END2END:
      # Save the exported checkpoint (used for downstreaming tasks) after every
      # 'checkpoint_interval' steps. And only export checkpoints after entering
      # e2e distillation training stage.
      if self.exported_ckpt_manager is None:
        self.build_exported_ckpt_manager()
      self.exported_ckpt_manager.save(
          checkpoint_number=self.current_step.numpy(),
          check_interval=True)

    return {
        'feature_transfer_mse':
            self.train_metrics['feature_transfer_mse'].result(),
        'beta_transfer_loss':
            self.train_metrics['beta_transfer_loss'].result(),
        'gamma_transfer_loss':
            self.train_metrics['gamma_transfer_loss'].result(),
        'attention_transfer_loss':
            self.train_metrics['attention_transfer_loss'].result(),
        'total_loss':
            self.train_metrics['total_loss'].result(),
        'lm_example_loss':
            self.train_metrics['lm_example_loss'].result(),
        'next_sentence_loss':
            self.train_metrics['next_sentence_loss'].result(),
        'masked_lm_accuracy':
            self.train_metrics['masked_lm_accuracy'].result(),
        'next_sentence_accuracy':
            self.train_metrics['next_sentence_accuracy'].result(),
        'learning_rate':
            self.current_optimizer.learning_rate(
                self.current_optimizer.iterations),
        'current_step':
            self.current_step,
        'optimizer_step':
            self.current_optimizer.iterations,
    }

  # TODO(longy): We only run evaluation on downstream tasks.
  def eval_begin(self):
    self.eval_metrics['masked_lm_accuracy'].reset_states()
    self.eval_metrics['next_sentence_accuracy'].reset_states()

  def eval_step(self, iterator):

    def step_fn(inputs):
      outputs = self.model(inputs, training=False)
      self.calculate_accuracy_metrics(inputs, outputs, self.eval_metrics)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    return {'masked_lm_accuracy':
                self.eval_metrics['masked_lm_accuracy'].result(),
            'next_sentence_accuracy':
                self.eval_metrics['next_sentence_accuracy'].result()}
