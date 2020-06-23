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
"""Tagging (e.g., NER/POS) task."""
import logging
import dataclasses
import tensorflow as tf
import tensorflow_hub as hub

from official.core import base_task
from official.modeling.hyperparams import config_definitions as cfg
from official.nlp.configs import encoders
from official.nlp.data import tagging_data_loader
from official.nlp.modeling import models
from official.nlp.tasks import utils


@dataclasses.dataclass
class TaggingConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  network: encoders.TransformerEncoderConfig = (
      encoders.TransformerEncoderConfig())
  num_classes: int = 0
  # The ignored label id will not contribute to loss.
  # A word may be tokenized into multiple word_pieces tokens, and we usually
  # assign the real label id for the first token of the word, and
  # `ignore_label_id` for the remaining tokens.
  ignore_label_id: int = 0
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()


@base_task.register_task_cls(TaggingConfig)
class TaggingTask(base_task.Task):
  """Task object for tagging (e.g., NER or POS)."""

  def __init__(self, params=cfg.TaskConfig):
    super(TaggingTask, self).__init__(params)
    if params.hub_module_url and params.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if params.num_classes == 0:
      raise ValueError('TaggingConfig.num_classes cannot be 0.')

    if params.hub_module_url:
      self._hub_module = hub.load(params.hub_module_url)
    else:
      self._hub_module = None

  def build_model(self):
    if self._hub_module:
      encoder_network = utils.get_encoder_from_hub(self._hub_module)
    else:
      encoder_network = encoders.instantiate_encoder_from_cfg(
          self.task_config.network)

    return models.BertTokenClassifier(
        network=encoder_network,
        num_classes=self.task_config.num_classes,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.task_config.network.initializer_range),
        dropout_rate=self.task_config.network.dropout_rate,
        output='logits')

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    model_outputs = tf.cast(model_outputs, tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, model_outputs, from_logits=True)
    # `ignore_label_id` will not contribute to loss.
    label_weights = tf.cast(
        tf.not_equal(labels, self.task_config.ignore_label_id),
        dtype=tf.float32)
    numerator_loss = tf.reduce_sum(loss * label_weights)
    denominator_loss = tf.reduce_sum(label_weights)
    loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
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
        y = tf.ones((1, params.seq_length), dtype=tf.int32)
        return (x, y)

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    dataset = tagging_data_loader.TaggingDataLoader(params).load(input_context)
    return dataset

  def build_metrics(self, training=None):
    del training
    # TODO(chendouble): evaluate using seqeval's f1/precision/recall.
    return [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]

  def process_metrics(self, metrics, labels, model_outputs):
    # `ignore_label_id` will not contribute to metrics.
    sample_weight = tf.cast(
        tf.not_equal(labels, self.task_config.ignore_label_id),
        dtype=tf.float32)
    for metric in metrics:
      metric.update_state(labels, model_outputs, sample_weight)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    # `ignore_label_id` will not contribute to metrics.
    sample_weight = tf.cast(
        tf.not_equal(labels, self.task_config.ignore_label_id),
        dtype=tf.float32)
    compiled_metrics.update_state(labels, model_outputs, sample_weight)

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
