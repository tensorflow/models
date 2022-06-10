# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Classifcation Task Showcase."""

import dataclasses
from typing import List, Mapping, Text
from seqeval import metrics as seqeval_metrics
import tensorflow as tf

from official.core import base_task
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.modeling import tf_utils
from official.modeling.hyperparams import base_config
from official.nlp.configs import encoders
from official.nlp.modeling import models
from official.nlp.tasks import utils
from official.projects.text_classification_example import classification_data_loader


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base span labeler configuration."""
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  head_dropout: float = 0.1
  head_initializer_range: float = 0.02


@dataclasses.dataclass
class ClassificationExampleConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  model: ModelConfig = ModelConfig()

  num_classes = 2
  class_names = ['A', 'B']
  train_data: cfg.DataConfig = classification_data_loader.ClassificationExampleDataConfig(
  )
  validation_data: cfg.DataConfig = classification_data_loader.ClassificationExampleDataConfig(
  )


class ClassificationExampleTask(base_task.Task):
  """Task object for classification."""

  def build_model(self) -> tf.keras.Model:
    if self.task_config.hub_module_url and self.task_config.init_checkpoint:
      raise ValueError('At most one of `hub_module_url` and '
                       '`init_checkpoint` can be specified.')
    if self.task_config.hub_module_url:
      encoder_network = utils.get_encoder_from_hub(
          self.task_config.hub_module_url)
    else:
      encoder_network = encoders.build_encoder(self.task_config.model.encoder)

    return models.BertClassifier(
        network=encoder_network,
        num_classes=len(self.task_config.class_names),
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self.task_config.model.head_initializer_range),
        dropout_rate=self.task_config.model.head_dropout)

  def build_losses(self, labels, model_outputs, aux_losses=None) -> tf.Tensor:
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, tf.cast(model_outputs, tf.float32), from_logits=True)
    return tf_utils.safe_mean(loss)

  def build_inputs(self,
                   params: cfg.DataConfig,
                   input_context=None) -> tf.data.Dataset:
    """Returns tf.data.Dataset for sentence_prediction task."""
    loader = classification_data_loader.ClassificationDataLoader(params)
    return loader.load(input_context)

  def inference_step(self, inputs,
                     model: tf.keras.Model) -> Mapping[str, tf.Tensor]:
    """Performs the forward step."""
    logits = model(inputs, training=False)
    return {
        'logits': logits,
        'predict_ids': tf.argmax(logits, axis=-1, output_type=tf.int32)
    }

  def validation_step(self,
                      inputs,
                      model: tf.keras.Model,
                      metrics=None) -> Mapping[str, tf.Tensor]:
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
    loss = self.build_losses(labels=labels, model_outputs=outputs['logits'])

    # Negative label ids are padding labels which should be ignored.
    real_label_index = tf.where(tf.greater_equal(labels, 0))
    predict_ids = tf.gather_nd(outputs['predict_ids'], real_label_index)
    label_ids = tf.gather_nd(labels, real_label_index)
    return {
        self.loss: loss,
        'predict_ids': predict_ids,
        'label_ids': label_ids,
    }

  def aggregate_logs(self,
                     state=None,
                     step_outputs=None) -> Mapping[Text, List[List[Text]]]:
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

  def reduce_aggregated_logs(self,
                             aggregated_logs,
                             global_step=None) -> Mapping[Text, float]:
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


@exp_factory.register_config_factory('example_bert_classification_example')
def bert_classification_example() -> cfg.ExperimentConfig:
  """Return a minimum experiment config for Bert token classification."""
  return cfg.ExperimentConfig(
      task=ClassificationExampleConfig(),
      trainer=cfg.TrainerConfig(
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'adamw',
              },
              'learning_rate': {
                  'type': 'polynomial',
              },
              'warmup': {
                  'type': 'polynomial'
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
