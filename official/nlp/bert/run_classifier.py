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

"""BERT classification or regression finetuning runner in TF 2.x."""

import functools
import json
import math
import os

# Import libraries
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from official.common import distribute_utils
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.utils.misc import keras_utils

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only', 'predict'],
    'One of {"train_and_eval", "export_only", "predict"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`. `predict`: takes a checkpoint and '
    'restores the model to output predictions on the test set.')
flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('train_data_size', None, 'Number of training samples '
                     'to use. If None, uses the full train data. '
                     '(default: None).')
flags.DEFINE_string('predict_checkpoint_path', None,
                    'Path to the checkpoint for predictions.')
flags.DEFINE_integer(
    'num_eval_per_epoch', 1,
    'Number of evaluations per epoch. The purpose of this flag is to provide '
    'more granular evaluation scores and checkpoints. For example, if original '
    'data has N samples and num_eval_per_epoch is n, then each epoch will be '
    'evaluated every N/n samples.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS

LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}


def get_loss_fn(num_classes):
  """Gets the classification loss function."""

  def classification_loss_fn(labels, logits):
    """Classification loss."""
    labels = tf.squeeze(labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(
        tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(
        tf.cast(one_hot_labels, dtype=tf.float32) * log_probs, axis=-1)
    return tf.reduce_mean(per_example_loss)

  return classification_loss_fn


def get_dataset_fn(input_file_pattern,
                   max_seq_length,
                   global_batch_size,
                   is_training,
                   label_type=tf.int64,
                   include_sample_weights=False,
                   num_samples=None):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_classifier_dataset(
        tf.io.gfile.glob(input_file_pattern),
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx,
        label_type=label_type,
        include_sample_weights=include_sample_weights,
        num_samples=num_samples)
    return dataset

  return _dataset_fn


def run_bert_classifier(strategy,
                        bert_config,
                        input_meta_data,
                        model_dir,
                        epochs,
                        steps_per_epoch,
                        steps_per_loop,
                        eval_steps,
                        warmup_steps,
                        initial_lr,
                        init_checkpoint,
                        train_input_fn,
                        eval_input_fn,
                        training_callbacks=True,
                        custom_callbacks=None,
                        custom_metrics=None):
  """Run BERT classifier training using low-level API."""
  max_seq_length = input_meta_data['max_seq_length']
  num_classes = input_meta_data.get('num_labels', 1)
  is_regression = num_classes == 1

  def _get_classifier_model():
    """Gets a classifier model."""
    classifier_model, core_model = (
        bert_models.classifier_model(
            bert_config,
            num_classes,
            max_seq_length,
            hub_module_url=FLAGS.hub_module_url,
            hub_module_trainable=FLAGS.hub_module_trainable))
    optimizer = optimization.create_optimizer(initial_lr,
                                              steps_per_epoch * epochs,
                                              warmup_steps, FLAGS.end_lr,
                                              FLAGS.optimizer_type)
    classifier_model.optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=common_flags.use_float16(),
        use_graph_rewrite=common_flags.use_graph_rewrite())
    return classifier_model, core_model

  # tf.keras.losses objects accept optional sample_weight arguments (eg. coming
  # from the dataset) to compute weighted loss, as used for the regression
  # tasks. The classification tasks, using the custom get_loss_fn don't accept
  # sample weights though.
  loss_fn = (tf.keras.losses.MeanSquaredError() if is_regression
             else get_loss_fn(num_classes))

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  if custom_metrics:
    metric_fn = custom_metrics
  elif is_regression:
    metric_fn = functools.partial(
        tf.keras.metrics.MeanSquaredError,
        'mean_squared_error',
        dtype=tf.float32)
  else:
    metric_fn = functools.partial(
        tf.keras.metrics.SparseCategoricalAccuracy,
        'accuracy',
        dtype=tf.float32)

  # Start training using Keras compile/fit API.
  logging.info('Training using TF 2.x Keras compile/fit API with '
               'distribution strategy.')
  return run_keras_compile_fit(
      model_dir,
      strategy,
      _get_classifier_model,
      train_input_fn,
      eval_input_fn,
      loss_fn,
      metric_fn,
      init_checkpoint,
      epochs,
      steps_per_epoch,
      steps_per_loop,
      eval_steps,
      training_callbacks=training_callbacks,
      custom_callbacks=custom_callbacks)


def run_keras_compile_fit(model_dir,
                          strategy,
                          model_fn,
                          train_input_fn,
                          eval_input_fn,
                          loss_fn,
                          metric_fn,
                          init_checkpoint,
                          epochs,
                          steps_per_epoch,
                          steps_per_loop,
                          eval_steps,
                          training_callbacks=True,
                          custom_callbacks=None):
  """Runs BERT classifier model using Keras compile/fit API."""

  with strategy.scope():
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn() if eval_input_fn else None
    bert_model, sub_model = model_fn()
    optimizer = bert_model.optimizer

    if init_checkpoint:
      checkpoint = tf.train.Checkpoint(model=sub_model, encoder=sub_model)
      checkpoint.read(init_checkpoint).assert_existing_objects_matched()

    if not isinstance(metric_fn, (list, tuple)):
      metric_fn = [metric_fn]
    bert_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[fn() for fn in metric_fn],
        steps_per_execution=steps_per_loop)

    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint = tf.train.Checkpoint(model=bert_model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=None,
        step_counter=optimizer.iterations,
        checkpoint_interval=0)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    if training_callbacks:
      if custom_callbacks is not None:
        custom_callbacks += [summary_callback, checkpoint_callback]
      else:
        custom_callbacks = [summary_callback, checkpoint_callback]

    history = bert_model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=eval_steps,
        callbacks=custom_callbacks)
    stats = {'total_training_steps': steps_per_epoch * epochs}
    if 'loss' in history.history:
      stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
      stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return bert_model, stats


def get_predictions_and_labels(strategy,
                               trained_model,
                               eval_input_fn,
                               is_regression=False,
                               return_probs=False):
  """Obtains predictions of trained model on evaluation data.

  Note that list of labels is returned along with the predictions because the
  order changes on distributing dataset over TPU pods.

  Args:
    strategy: Distribution strategy.
    trained_model: Trained model with preloaded weights.
    eval_input_fn: Input function for evaluation data.
    is_regression: Whether it is a regression task.
    return_probs: Whether to return probabilities of classes.

  Returns:
    predictions: List of predictions.
    labels: List of gold labels corresponding to predictions.
  """

  @tf.function
  def test_step(iterator):
    """Computes predictions on distributed devices."""

    def _test_step_fn(inputs):
      """Replicated predictions."""
      inputs, labels = inputs
      logits = trained_model(inputs, training=False)
      if not is_regression:
        probabilities = tf.nn.softmax(logits)
        return probabilities, labels
      else:
        return logits, labels

    outputs, labels = strategy.run(_test_step_fn, args=(next(iterator),))
    # outputs: current batch logits as a tuple of shard logits
    outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                    outputs)
    labels = tf.nest.map_structure(strategy.experimental_local_results, labels)
    return outputs, labels

  def _run_evaluation(test_iterator):
    """Runs evaluation steps."""
    preds, golds = list(), list()
    try:
      with tf.experimental.async_scope():
        while True:
          probabilities, labels = test_step(test_iterator)
          for cur_probs, cur_labels in zip(probabilities, labels):
            if return_probs:
              preds.extend(cur_probs.numpy().tolist())
            else:
              preds.extend(tf.math.argmax(cur_probs, axis=1).numpy())
            golds.extend(cur_labels.numpy().tolist())
    except (StopIteration, tf.errors.OutOfRangeError):
      tf.experimental.async_clear_error()
    return preds, golds

  test_iter = iter(strategy.distribute_datasets_from_function(eval_input_fn))
  predictions, labels = _run_evaluation(test_iter)

  return predictions, labels


def export_classifier(model_export_path, input_meta_data, bert_config,
                      model_dir):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.
    bert_config: Bert configuration file to define core bert layers.
    model_dir: The directory where the model weights and training/evaluation
      summaries are stored.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  if not model_dir:
    raise ValueError('Export path is not specified: %s' % model_dir)

  # Export uses float32 for now, even if training uses mixed precision.
  tf.keras.mixed_precision.set_global_policy('float32')
  classifier_model = bert_models.classifier_model(
      bert_config,
      input_meta_data.get('num_labels', 1),
      hub_module_url=FLAGS.hub_module_url,
      hub_module_trainable=False)[0]

  model_saving_utils.export_bert_model(
      model_export_path, model=classifier_model, checkpoint_dir=model_dir)


def run_bert(strategy,
             input_meta_data,
             model_config,
             train_input_fn=None,
             eval_input_fn=None,
             init_checkpoint=None,
             custom_callbacks=None,
             custom_metrics=None):
  """Run BERT training."""
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_session_config(FLAGS.enable_xla)
  performance.set_mixed_precision_policy(common_flags.dtype())

  epochs = FLAGS.num_train_epochs * FLAGS.num_eval_per_epoch
  train_data_size = (
      input_meta_data['train_data_size'] // FLAGS.num_eval_per_epoch)
  if FLAGS.train_data_size:
    train_data_size = min(train_data_size, FLAGS.train_data_size)
    logging.info('Updated train_data_size: %s', train_data_size)
  steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
  warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
  eval_steps = int(
      math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))

  if not strategy:
    raise ValueError('Distribution strategy has not been specified.')

  if not custom_callbacks:
    custom_callbacks = []

  if FLAGS.log_steps:
    custom_callbacks.append(
        keras_utils.TimeHistory(
            batch_size=FLAGS.train_batch_size,
            log_steps=FLAGS.log_steps,
            logdir=FLAGS.model_dir))

  trained_model, _ = run_bert_classifier(
      strategy,
      model_config,
      input_meta_data,
      FLAGS.model_dir,
      epochs,
      steps_per_epoch,
      FLAGS.steps_per_loop,
      eval_steps,
      warmup_steps,
      FLAGS.learning_rate,
      init_checkpoint or FLAGS.init_checkpoint,
      train_input_fn,
      eval_input_fn,
      custom_callbacks=custom_callbacks,
      custom_metrics=custom_metrics)

  if FLAGS.model_export_path:
    model_saving_utils.export_bert_model(
        FLAGS.model_export_path, model=trained_model)
  return trained_model


def custom_main(custom_callbacks=None, custom_metrics=None):
  """Run classification or regression.

  Args:
    custom_callbacks: list of tf.keras.Callbacks passed to training loop.
    custom_metrics: list of metrics passed to the training loop.
  """
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))
  label_type = LABEL_TYPES_MAP[input_meta_data.get('label_type', 'int')]
  include_sample_weights = input_meta_data.get('has_sample_weights', False)

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.mode == 'export_only':
    export_classifier(FLAGS.model_export_path, input_meta_data, bert_config,
                      FLAGS.model_dir)
    return

  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  eval_input_fn = get_dataset_fn(
      FLAGS.eval_data_path,
      input_meta_data['max_seq_length'],
      FLAGS.eval_batch_size,
      is_training=False,
      label_type=label_type,
      include_sample_weights=include_sample_weights)

  if FLAGS.mode == 'predict':
    num_labels = input_meta_data.get('num_labels', 1)
    with strategy.scope():
      classifier_model = bert_models.classifier_model(
          bert_config, num_labels)[0]
      checkpoint = tf.train.Checkpoint(model=classifier_model)
      latest_checkpoint_file = (
          FLAGS.predict_checkpoint_path or
          tf.train.latest_checkpoint(FLAGS.model_dir))
      assert latest_checkpoint_file
      logging.info('Checkpoint file %s found and restoring from '
                   'checkpoint', latest_checkpoint_file)
      checkpoint.restore(
          latest_checkpoint_file).assert_existing_objects_matched()
      preds, _ = get_predictions_and_labels(
          strategy,
          classifier_model,
          eval_input_fn,
          is_regression=(num_labels == 1),
          return_probs=True)
    output_predict_file = os.path.join(FLAGS.model_dir, 'test_results.tsv')
    with tf.io.gfile.GFile(output_predict_file, 'w') as writer:
      logging.info('***** Predict results *****')
      for probabilities in preds:
        output_line = '\t'.join(
            str(class_probability)
            for class_probability in probabilities) + '\n'
        writer.write(output_line)
    return

  if FLAGS.mode != 'train_and_eval':
    raise ValueError('Unsupported mode is specified: %s' % FLAGS.mode)
  train_input_fn = get_dataset_fn(
      FLAGS.train_data_path,
      input_meta_data['max_seq_length'],
      FLAGS.train_batch_size,
      is_training=True,
      label_type=label_type,
      include_sample_weights=include_sample_weights,
      num_samples=FLAGS.train_data_size)
  run_bert(
      strategy,
      input_meta_data,
      bert_config,
      train_input_fn,
      eval_input_fn,
      custom_callbacks=custom_callbacks,
      custom_metrics=custom_metrics)


def main(_):
  custom_main(custom_callbacks=None, custom_metrics=None)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
