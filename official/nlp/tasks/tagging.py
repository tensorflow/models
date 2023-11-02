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

"""Tagging (e.g., NER/POS) task."""
from typing import List, Optional, Tuple

import dataclasses
import orbit

from seqeval import metrics as seqeval_metrics

import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.data import data_loader_factory
from official.nlp.modeling import models
from official.nlp.tasks import utils


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base span labeler configuration."""
  encoder: encoders.EncoderConfig = dataclasses.field(default_factory=encoders.EncoderConfig)
  head_dropout: float = 0.1
  head_initializer_range: float = 0.02


@dataclasses.dataclass
class TaggingConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

  # The real class names, the order of which should match real label id.
  # Note that a word may be tokenized into multiple word_pieces tokens, and
  # we asssume the real label id (non-negative) is assigned to the first token
  # of the word, and a negative label id is assigned to the remaining tokens.
  # The negative label id will not contribute to loss and metrics.
  class_names: Optional[List[str]] = None
  train_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)
  validation_data: cfg.DataConfig = dataclasses.field(default_factory=cfg.DataConfig)


def _masked_labels_and_weights(y_true):
  """Masks negative values from token level labels.

  Args:
    y_true: Token labels, typically shape (batch_size, seq_len), where tokens
      with negative labels should be ignored during loss/accuracy calculation.

  Returns:
    (masked_y_true, masked_weights) where `masked_y_true` is the input
    with each negative label replaced with zero and `masked_weights` is 0.0
    where negative labels were replaced and 1.0 for original labels.
  """
  # Ignore the classes of tokens with negative values.
  mask = tf.greater_equal(y_true, 0)
  # Replace negative labels, which are out of bounds for some loss functions,
  # with zero.
  masked_y_true = tf.where(mask, y_true, 0)
  return masked_y_true, tf.cast(mask, tf.float32)


@task_factory.register_task_cls(TaggingConfig)
class TaggingTask(base_task.Task):
  """Task object for tagging (e.g., NER or POS)."""

  def build_model(self):
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url)
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)

    return models.BertTokenClassifier(
        network=encoder_network,
        num_classes=len(self.task_config.class_names),
        initializer=tf_keras.initializers.TruncatedNormal(
            stddev=self.task_config.model.head_initializer_range),
        dropout_rate=self.task_config.model.head_dropout,
        output='logits',
        output_encoder_outputs=True)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    logits = tf.cast(model_outputs['logits'], tf.float32)
    masked_labels, masked_weights = _masked_labels_and_weights(labels)
    loss = tf_keras.losses.sparse_categorical_crossentropy(
        masked_labels, logits, from_logits=True)
    numerator_loss = tf.reduce_sum(loss * masked_weights)
    denominator_loss = tf.reduce_sum(masked_weights)
    loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)
    return loss

  def build_inputs(self, params: cfg.DataConfig, input_context=None):
    """Returns tf.data.Dataset for sentence_prediction task."""
    if params.input_path == 'dummy':

      def dummy_data(_):
        dummy_ids = tf.zeros((1, params.seq_length), dtype=tf.int32)
        x = dict(
            input_word_ids=dummy_ids,
            input_mask=dummy_ids,
            input_type_ids=dummy_ids)

        # Include some label_id as -1, which will be ignored in loss/metrics.
        y = tf.random.uniform(
            shape=(1, params.seq_length),
            minval=-1,
            maxval=len(self.task_config.class_names),
            dtype=tf.dtypes.int32)
        return (x, y)

      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(
          dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      return dataset

    return data_loader_factory.get_data_loader(params).load(input_context)

  def inference_step(self, inputs, model: tf_keras.Model):
    """Performs the forward step."""
    logits = model(inputs, training=False)['logits']
    return {'logits': logits,
            'predict_ids': tf.argmax(logits, axis=-1, output_type=tf.int32)}

  def validation_step(self, inputs, model: tf_keras.Model, metrics=None):
    """Validatation step.

    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    features, labels = inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(labels=labels, model_outputs=outputs)

    # Negative label ids are padding labels which should be ignored.
    real_label_index = tf.where(tf.greater_equal(labels, 0))
    predict_ids = tf.gather_nd(outputs['predict_ids'], real_label_index)
    label_ids = tf.gather_nd(labels, real_label_index)
    return {
        self.loss: loss,
        'predict_ids': predict_ids,
        'label_ids': label_ids,
    }

  def aggregate_logs(self, state=None, step_outputs=None):
    """Aggregates over logs returned from a validation step."""
    if state is None:
      state = {'predict_class': [], 'label_class': []}

    def id_to_class_name(batched_ids):
      class_names = []
      for per_example_ids in batched_ids:
        class_names.append([])
        for per_token_id in per_example_ids.numpy().tolist():
          class_names[-1].append(self.task_config.class_names[per_token_id])

      return class_names

    # Convert id to class names, because `seqeval_metrics` relies on the class
    # name to decide IOB tags.
    state['predict_class'].extend(id_to_class_name(step_outputs['predict_ids']))
    state['label_class'].extend(id_to_class_name(step_outputs['label_ids']))
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    """Reduces aggregated logs over validation steps."""
    label_class = aggregated_logs['label_class']
    predict_class = aggregated_logs['predict_class']
    return {
        'f1':
            seqeval_metrics.f1_score(label_class, predict_class),
        'precision':
            seqeval_metrics.precision_score(label_class, predict_class),
        'recall':
            seqeval_metrics.recall_score(label_class, predict_class),
        'accuracy':
            seqeval_metrics.accuracy_score(label_class, predict_class),
    }


def predict(task: TaggingTask,
            params: cfg.DataConfig,
            model: tf_keras.Model) -> List[Tuple[int, int, List[int]]]:
  """Predicts on the input data.

  Args:
    task: A `TaggingTask` object.
    params: A `cfg.DataConfig` object.
    model: A keras.Model.

  Returns:
    A list of tuple. Each tuple contains `sentence_id`, `sub_sentence_id` and
      a list of predicted ids.
  """

  def predict_step(inputs):
    """Replicated prediction calculation."""
    x, y = inputs
    sentence_ids = x.pop('sentence_id')
    sub_sentence_ids = x.pop('sub_sentence_id')
    outputs = task.inference_step(x, model)
    predict_ids = outputs['predict_ids']
    label_mask = tf.greater_equal(y, 0)
    return dict(
        predict_ids=predict_ids,
        label_mask=label_mask,
        sentence_ids=sentence_ids,
        sub_sentence_ids=sub_sentence_ids)

  def aggregate_fn(state, outputs):
    """Concatenates model's outputs."""
    if state is None:
      state = []

    for (batch_predict_ids, batch_label_mask, batch_sentence_ids,
         batch_sub_sentence_ids) in zip(outputs['predict_ids'],
                                        outputs['label_mask'],
                                        outputs['sentence_ids'],
                                        outputs['sub_sentence_ids']):
      for (tmp_predict_ids, tmp_label_mask, tmp_sentence_id,
           tmp_sub_sentence_id) in zip(batch_predict_ids.numpy(),
                                       batch_label_mask.numpy(),
                                       batch_sentence_ids.numpy(),
                                       batch_sub_sentence_ids.numpy()):
        real_predict_ids = []
        assert len(tmp_predict_ids) == len(tmp_label_mask)
        for i in range(len(tmp_predict_ids)):
          # Skip the padding label.
          if tmp_label_mask[i]:
            real_predict_ids.append(tmp_predict_ids[i])
        state.append((tmp_sentence_id, tmp_sub_sentence_id, real_predict_ids))

    return state

  dataset = orbit.utils.make_distributed_dataset(tf.distribute.get_strategy(),
                                                 task.build_inputs, params)
  outputs = utils.predict(predict_step, aggregate_fn, dataset)
  return sorted(outputs, key=lambda x: (x[0], x[1]))
