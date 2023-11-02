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

"""The helper for finetuning binaries."""
import json
import math
import sys
from typing import Any, Dict, List, Optional

from absl import logging
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.modeling import hyperparams
from official.nlp.configs import encoders
from official.nlp.data import question_answering_dataloader
from official.nlp.data import sentence_prediction_dataloader
from official.nlp.data import tagging_dataloader
from official.nlp.tasks import question_answering
from official.nlp.tasks import sentence_prediction
from official.nlp.tasks import tagging


def override_trainer_cfg(trainer_cfg: cfg.TrainerConfig, learning_rate: float,
                         num_epoch: int, global_batch_size: int,
                         warmup_ratio: float, training_data_size: int,
                         eval_data_size: int, num_eval_per_epoch: int,
                         best_checkpoint_export_subdir: str,
                         best_checkpoint_eval_metric: str,
                         best_checkpoint_metric_comp: str):
  """Overrides a `cfg.TrainerConfig` object."""
  steps_per_epoch = training_data_size // global_batch_size
  train_steps = steps_per_epoch * num_epoch
  # TODO(b/165081095): always set to -1 after the bug is resolved.
  if eval_data_size:
    eval_steps = int(math.ceil(eval_data_size / global_batch_size))
  else:
    eval_steps = -1  # exhaust the validation data.
  warmp_steps = int(train_steps * warmup_ratio)
  validation_interval = steps_per_epoch // num_eval_per_epoch
  trainer_cfg.override({
      'optimizer_config': {
          'learning_rate': {
              'type': 'polynomial',
              'polynomial': {
                  'decay_steps': train_steps,
                  'initial_learning_rate': learning_rate,
                  'end_learning_rate': 0,
              }
          },
          'optimizer': {
              'type': 'adamw',
          },
          'warmup': {
              'polynomial': {
                  'warmup_steps': warmp_steps,
              },
              'type': 'polynomial',
          },
      },
      'train_steps': train_steps,
      'validation_interval': validation_interval,
      'validation_steps': eval_steps,
      'best_checkpoint_export_subdir': best_checkpoint_export_subdir,
      'best_checkpoint_eval_metric': best_checkpoint_eval_metric,
      'best_checkpoint_metric_comp': best_checkpoint_metric_comp,
  })


def load_model_config_file(model_config_file: str) -> Dict[str, Any]:
  """Loads bert config json file or `encoders.EncoderConfig` in yaml file."""
  if not model_config_file:
    # model_config_file may be empty when using tf.hub.
    return {}

  try:
    encoder_config = encoders.EncoderConfig()
    encoder_config = hyperparams.override_params_dict(
        encoder_config, model_config_file, is_strict=True)
    logging.info('Load encoder_config yaml file from %s.', model_config_file)
    return encoder_config.as_dict()
  except KeyError:
    pass

  logging.info('Load bert config json file from %s', model_config_file)
  with tf.io.gfile.GFile(model_config_file, 'r') as reader:
    text = reader.read()
    config = json.loads(text)

  def get_value(key1, key2):
    if key1 in config and key2 in config:
      raise ValueError('Unexpected that both %s and %s are in config.' %
                       (key1, key2))

    return config[key1] if key1 in config else config[key2]

  def get_value_or_none(key):
    return config[key] if key in config else None

  # Support both legacy bert_config attributes and the new config attributes.
  return {
      'bert': {
          'attention_dropout_rate':
              get_value('attention_dropout_rate',
                        'attention_probs_dropout_prob'),
          'dropout_rate':
              get_value('dropout_rate', 'hidden_dropout_prob'),
          'hidden_activation':
              get_value('hidden_activation', 'hidden_act'),
          'hidden_size':
              config['hidden_size'],
          'embedding_size':
              get_value_or_none('embedding_size'),
          'initializer_range':
              config['initializer_range'],
          'intermediate_size':
              config['intermediate_size'],
          'max_position_embeddings':
              config['max_position_embeddings'],
          'num_attention_heads':
              config['num_attention_heads'],
          'num_layers':
              get_value('num_layers', 'num_hidden_layers'),
          'type_vocab_size':
              config['type_vocab_size'],
          'vocab_size':
              config['vocab_size'],
      }
  }


def override_sentence_prediction_task_config(
    task_cfg: sentence_prediction.SentencePredictionConfig,
    model_config_file: str,
    init_checkpoint: str,
    hub_module_url: str,
    global_batch_size: int,
    train_input_path: str,
    validation_input_path: str,
    seq_length: int,
    num_classes: int,
    metric_type: Optional[str] = 'accuracy',
    label_type: Optional[str] = 'int'):
  """Overrides a `SentencePredictionConfig` object."""
  task_cfg.override({
      'init_checkpoint': init_checkpoint,
      'metric_type': metric_type,
      'model': {
          'num_classes': num_classes,
          'encoder': load_model_config_file(model_config_file),
      },
      'hub_module_url': hub_module_url,
      'train_data': {
          'drop_remainder': True,
          'global_batch_size': global_batch_size,
          'input_path': train_input_path,
          'is_training': True,
          'seq_length': seq_length,
          'label_type': label_type,
      },
      'validation_data': {
          'drop_remainder': False,
          'global_batch_size': global_batch_size,
          'input_path': validation_input_path,
          'is_training': False,
          'seq_length': seq_length,
          'label_type': label_type,
      }
  })


def override_qa_task_config(
    task_cfg: question_answering.QuestionAnsweringConfig,
    model_config_file: str, init_checkpoint: str, hub_module_url: str,
    global_batch_size: int, train_input_path: str, validation_input_path: str,
    seq_length: int, tokenization: str, vocab_file: str, do_lower_case: bool,
    version_2_with_negative: bool):
  """Overrides a `QuestionAnsweringConfig` object."""
  task_cfg.override({
      'init_checkpoint': init_checkpoint,
      'model': {
          'encoder': load_model_config_file(model_config_file),
      },
      'hub_module_url': hub_module_url,
      'train_data': {
          'drop_remainder': True,
          'global_batch_size': global_batch_size,
          'input_path': train_input_path,
          'is_training': True,
          'seq_length': seq_length,
      },
      'validation_data': {
          'do_lower_case': do_lower_case,
          'drop_remainder': False,
          'global_batch_size': global_batch_size,
          'input_path': validation_input_path,
          'is_training': False,
          'seq_length': seq_length,
          'tokenization': tokenization,
          'version_2_with_negative': version_2_with_negative,
          'vocab_file': vocab_file,
      }
  })


def override_tagging_task_config(task_cfg: tagging.TaggingConfig,
                                 model_config_file: str, init_checkpoint: str,
                                 hub_module_url: str, global_batch_size: int,
                                 train_input_path: str,
                                 validation_input_path: str, seq_length: int,
                                 class_names: List[str]):
  """Overrides a `TaggingConfig` object."""
  task_cfg.override({
      'init_checkpoint': init_checkpoint,
      'model': {
          'encoder': load_model_config_file(model_config_file),
      },
      'hub_module_url': hub_module_url,
      'train_data': {
          'drop_remainder': True,
          'global_batch_size': global_batch_size,
          'input_path': train_input_path,
          'is_training': True,
          'seq_length': seq_length,
      },
      'validation_data': {
          'drop_remainder': False,
          'global_batch_size': global_batch_size,
          'input_path': validation_input_path,
          'is_training': False,
          'seq_length': seq_length,
      },
      'class_names': class_names,
  })


def write_glue_classification(task,
                              model,
                              input_file,
                              output_file,
                              predict_batch_size,
                              seq_length,
                              class_names,
                              label_type='int',
                              min_float_value=None,
                              max_float_value=None):
  """Makes classification predictions for glue and writes to output file.

  Args:
    task: `Task` instance.
    model: `keras.Model` instance.
    input_file: Input test data file path.
    output_file: Output test data file path.
    predict_batch_size: Batch size for prediction.
    seq_length: Input sequence length.
    class_names: List of string class names.
    label_type: String denoting label type ('int', 'float'), defaults to 'int'.
    min_float_value: If set, predictions will be min-clipped to this value (only
      for regression when `label_type` is set to 'float'). Defaults to `None`
      (no clipping).
    max_float_value: If set, predictions will be max-clipped to this value (only
      for regression when `label_type` is set to 'float'). Defaults to `None`
      (no clipping).
  """
  if label_type not in ('int', 'float'):
    raise ValueError('Unsupported `label_type`. Given: %s, expected `int` or '
                     '`float`.' % label_type)

  data_config = sentence_prediction_dataloader.SentencePredictionDataConfig(
      input_path=input_file,
      global_batch_size=predict_batch_size,
      is_training=False,
      seq_length=seq_length,
      label_type=label_type,
      drop_remainder=False,
      include_example_id=True)
  predictions = sentence_prediction.predict(task, data_config, model)

  if label_type == 'float':
    min_float_value = (-sys.float_info.max
                       if min_float_value is None else min_float_value)
    max_float_value = (
        sys.float_info.max if max_float_value is None else max_float_value)

    # Clip predictions to range [min_float_value, max_float_value].
    predictions = [
        min(max(prediction, min_float_value), max_float_value)
        for prediction in predictions
    ]

  with tf.io.gfile.GFile(output_file, 'w') as writer:
    writer.write('index\tprediction\n')
    for index, prediction in enumerate(predictions):
      if label_type == 'float':
        # Regression.
        writer.write('%d\t%.3f\n' % (index, prediction))
      else:
        # Classification.
        writer.write('%d\t%s\n' % (index, class_names[prediction]))


def write_superglue_classification(task,
                                   model,
                                   input_file,
                                   output_file,
                                   predict_batch_size,
                                   seq_length,
                                   class_names,
                                   label_type='int'):
  """Makes classification predictions for superglue and writes to output file.

  Args:
    task: `Task` instance.
    model: `keras.Model` instance.
    input_file: Input test data file path.
    output_file: Output test data file path.
    predict_batch_size: Batch size for prediction.
    seq_length: Input sequence length.
    class_names: List of string class names.
    label_type: String denoting label type ('int', 'float'), defaults to 'int'.
  """
  if label_type not in 'int':
    raise ValueError('Unsupported `label_type`. Given: %s, expected `int` or '
                     '`float`.' % label_type)

  data_config = sentence_prediction_dataloader.SentencePredictionDataConfig(
      input_path=input_file,
      global_batch_size=predict_batch_size,
      is_training=False,
      seq_length=seq_length,
      label_type=label_type,
      drop_remainder=False,
      include_example_id=True)
  predictions = sentence_prediction.predict(task, data_config, model)

  with tf.io.gfile.GFile(output_file, 'w') as writer:
    for index, prediction in enumerate(predictions):
      if label_type == 'int':
        # Classification.
        writer.write('{"idx": %d, "label": %s}\n' %
                     (index, class_names[prediction]))


def write_xtreme_classification(task,
                                model,
                                input_file,
                                output_file,
                                predict_batch_size,
                                seq_length,
                                class_names,
                                translated_input_file=None,
                                test_time_aug_wgt=0.3):
  """Makes classification predictions for xtreme and writes to output file."""
  data_config = sentence_prediction_dataloader.SentencePredictionDataConfig(
      input_path=input_file,
      seq_length=seq_length,
      is_training=False,
      label_type='int',
      global_batch_size=predict_batch_size,
      drop_remainder=False,
      include_example_id=True)
  if translated_input_file is not None:
    data_config_aug = (
        sentence_prediction_dataloader.SentencePredictionDataConfig(
            input_path=translated_input_file,
            seq_length=seq_length,
            is_training=False,
            label_type='int',
            global_batch_size=predict_batch_size,
            drop_remainder=False,
            include_example_id=True))
  else:
    data_config_aug = None
  predictions = sentence_prediction.predict(task, data_config, model,
                                            data_config_aug, test_time_aug_wgt)
  with tf.io.gfile.GFile(output_file, 'w') as writer:
    for prediction in predictions:
      writer.write('%s\n' % class_names[prediction])


def write_question_answering(task,
                             model,
                             input_file,
                             output_file,
                             predict_batch_size,
                             seq_length,
                             tokenization,
                             vocab_file,
                             do_lower_case,
                             version_2_with_negative=False):
  """Makes question answering predictions and writes to output file."""
  data_config = question_answering_dataloader.QADataConfig(
      do_lower_case=do_lower_case,
      doc_stride=128,
      drop_remainder=False,
      global_batch_size=predict_batch_size,
      input_path=input_file,
      is_training=False,
      query_length=64,
      seq_length=seq_length,
      tokenization=tokenization,
      version_2_with_negative=version_2_with_negative,
      vocab_file=vocab_file)
  all_predictions, _, _ = question_answering.predict(task, data_config, model)
  with tf.io.gfile.GFile(output_file, 'w') as writer:
    writer.write(json.dumps(all_predictions, indent=4) + '\n')


def write_tagging(task, model, input_file, output_file, predict_batch_size,
                  seq_length):
  """Makes tagging predictions and writes to output file."""
  data_config = tagging_dataloader.TaggingDataConfig(
      input_path=input_file,
      is_training=False,
      seq_length=seq_length,
      global_batch_size=predict_batch_size,
      drop_remainder=False,
      include_sentence_id=True)
  results = tagging.predict(task, data_config, model)
  class_names = task.task_config.class_names
  last_sentence_id = -1

  with tf.io.gfile.GFile(output_file, 'w') as writer:
    for sentence_id, _, predict_ids in results:
      token_labels = [class_names[x] for x in predict_ids]
      assert sentence_id == last_sentence_id or (
          sentence_id == last_sentence_id + 1)

      if sentence_id != last_sentence_id and last_sentence_id != -1:
        writer.write('\n')

      writer.write('\n'.join(token_labels))
      writer.write('\n')
      last_sentence_id = sentence_id
