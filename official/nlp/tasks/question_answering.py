# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Question answering task."""
import logging
import dataclasses
import tensorflow as tf
import tensorflow_hub as hub

from official.core import base_task
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.bert import input_pipeline
from official.nlp.configs import encoders
from official.nlp.modeling import models
from official.nlp.tasks import utils


@dataclasses.dataclass
class QuestionAnsweringConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  model: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@base_task.register_task_cls(QuestionAnsweringConfig)
class QuestionAnsweringTask(base_task.Task):
  """Task object for question answering.

  TODO(lehou): Add post-processing.
  """

  def __init__(self, params=cfg.TaskConfig):
    super(QuestionAnsweringTask, self).__init__(params)
    if params.hub_module_url and params.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if params.hub_module_url:
      self._hub_module = hub.load(params.hub_module_url)
    else:
      self._hub_module = None

  def build_model(self):
    if self._hub_module:
      encoder_network = utils.get_encoder_from_hub(self._hub_module)
    else:
      encoder_network = encoders.instantiate_encoder_from_cfg(
          self.task_config.model)

    return models.BertSpanLabeler(
        network=encoder_network,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.task_config.model.initializer_range))

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    start_positions = labels['start_positions']
    end_positions = labels['end_positions']
    start_logits, end_logits = model_outputs

    start_loss = tf.keras.losses.sparse_categorical_crossentropy(
        start_positions,
        tf.cast(start_logits, dtype=tf.float32),
        from_logits=True)
    end_loss = tf.keras.losses.sparse_categorical_crossentropy(
        end_positions,
        tf.cast(end_logits, dtype=tf.float32),
        from_logits=True)

    loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    return loss

  def build_inputs(self, params, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':
      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids)
        y = dict(
            start_positions=tf.constant(0, dtype=tf.int32),
            end_positions=tf.constant(1, dtype=tf.int32))
        return (x, y)

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    batch_size = input_context.get_per_replica_batch_size(
        params.global_batch_size) if input_context else params.global_batch_size
    # TODO(chendouble): add and use nlp.data.question_answering_dataloader.
    dataset = input_pipeline.create_squad_dataset(
        params.input_path,
        params.seq_length,
        batch_size,
        is_training=params.is_training,
        input_pipeline_context=input_context)
    return dataset

  def build_metrics(self, training=None):
    del training
    # TODO(lehou): a list of metrics doesn't work the same as in compile/fit.
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(
            name='start_position_accuracy'),
        tf.keras.metrics.SparseCategoricalAccuracy(
            name='end_position_accuracy'),
    ]
    return metrics

  def process_metrics(self, metrics, labels, model_outputs):
    metrics = dict([(metric.name, metric) for metric in metrics])
    start_logits, end_logits = model_outputs
    metrics['start_position_accuracy'].update_state(
        labels['start_positions'], start_logits)
    metrics['end_position_accuracy'].update_state(
        labels['end_positions'], end_logits)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    start_logits, end_logits = model_outputs
    compiled_metrics.update_state(
        y_true=labels,  # labels has keys 'start_positions' and 'end_positions'.
        y_pred={'start_positions': start_logits, 'end_positions': end_logits})

  def initialize(self, model):
    """Load a pretrained checkpoint (if exists) and then train from iter 0."""
    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    ckpt = tf.train.Checkpoint(**model.checkpoint_items)
    status = ckpt.restore(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)
