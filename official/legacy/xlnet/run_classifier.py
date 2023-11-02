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

"""XLNet classification finetuning runner in tf2.0."""

import functools
# Import libraries
from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf, tf_keras
# pylint: disable=unused-import
from official.common import distribute_utils
from official.legacy.xlnet import common_flags
from official.legacy.xlnet import data_utils
from official.legacy.xlnet import optimization
from official.legacy.xlnet import training_utils
from official.legacy.xlnet import xlnet_config
from official.legacy.xlnet import xlnet_modeling as modeling

flags.DEFINE_integer("n_class", default=2, help="Number of classes.")
flags.DEFINE_string(
    "summary_type",
    default="last",
    help="Method used to summarize a sequence into a vector.")

FLAGS = flags.FLAGS


def get_classificationxlnet_model(model_config,
                                  run_config,
                                  n_class,
                                  summary_type="last"):
  model = modeling.ClassificationXLNetModel(
      model_config, run_config, n_class, summary_type, name="model")
  return model


def run_evaluation(strategy,
                   test_input_fn,
                   eval_steps,
                   model,
                   step,
                   eval_summary_writer=None):
  """Run evaluation for classification task.

  Args:
    strategy: distribution strategy.
    test_input_fn: input function for evaluation data.
    eval_steps: total number of evaluation steps.
    model: keras model object.
    step: current train step.
    eval_summary_writer: summary writer used to record evaluation metrics.  As
      there are fake data samples in validation set, we use mask to get rid of
      them when calculating the accuracy. For the reason that there will be
      dynamic-shape tensor, we first collect logits, labels and masks from TPU
      and calculate the accuracy via numpy locally.

  Returns:
    A float metric, accuracy.
  """

  def _test_step_fn(inputs):
    """Replicated validation step."""

    inputs["mems"] = None
    _, logits = model(inputs, training=False)
    return logits, inputs["label_ids"], inputs["is_real_example"]

  @tf.function
  def _run_evaluation(test_iterator):
    """Runs validation steps."""
    logits, labels, masks = strategy.run(
        _test_step_fn, args=(next(test_iterator),))
    return logits, labels, masks

  test_iterator = data_utils.get_input_iterator(test_input_fn, strategy)
  correct = 0
  total = 0
  for _ in range(eval_steps):
    logits, labels, masks = _run_evaluation(test_iterator)
    logits = strategy.experimental_local_results(logits)
    labels = strategy.experimental_local_results(labels)
    masks = strategy.experimental_local_results(masks)
    merged_logits = []
    merged_labels = []
    merged_masks = []

    for i in range(strategy.num_replicas_in_sync):
      merged_logits.append(logits[i].numpy())
      merged_labels.append(labels[i].numpy())
      merged_masks.append(masks[i].numpy())
    merged_logits = np.vstack(np.array(merged_logits))
    merged_labels = np.hstack(np.array(merged_labels))
    merged_masks = np.hstack(np.array(merged_masks))
    real_index = np.where(np.equal(merged_masks, 1))
    correct += np.sum(
        np.equal(
            np.argmax(merged_logits[real_index], axis=-1),
            merged_labels[real_index]))
    total += np.shape(real_index)[-1]
  accuracy = float(correct) / float(total)
  logging.info("Train step: %d  /  acc = %d/%d = %f", step, correct, total,
               accuracy)
  if eval_summary_writer:
    with eval_summary_writer.as_default():
      tf.summary.scalar("eval_acc", float(correct) / float(total), step=step)
      eval_summary_writer.flush()
  return accuracy


def get_metric_fn():
  train_acc_metric = tf_keras.metrics.SparseCategoricalAccuracy(
      "acc", dtype=tf.float32)
  return train_acc_metric


def main(unused_argv):
  del unused_argv
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.strategy_type,
      tpu_address=FLAGS.tpu)
  if strategy:
    logging.info("***** Number of cores used : %d",
                 strategy.num_replicas_in_sync)
  train_input_fn = functools.partial(data_utils.get_classification_input_data,
                                     FLAGS.train_batch_size, FLAGS.seq_len,
                                     strategy, True, FLAGS.train_tfrecord_path)
  test_input_fn = functools.partial(data_utils.get_classification_input_data,
                                    FLAGS.test_batch_size, FLAGS.seq_len,
                                    strategy, False, FLAGS.test_tfrecord_path)

  total_training_steps = FLAGS.train_steps
  steps_per_loop = FLAGS.iterations
  eval_steps = int(FLAGS.test_data_size / FLAGS.test_batch_size)
  eval_fn = functools.partial(run_evaluation, strategy, test_input_fn,
                              eval_steps)
  optimizer, learning_rate_fn = optimization.create_optimizer(
      FLAGS.learning_rate,
      total_training_steps,
      FLAGS.warmup_steps,
      adam_epsilon=FLAGS.adam_epsilon)
  model_config = xlnet_config.XLNetConfig(FLAGS)
  run_config = xlnet_config.create_run_config(True, False, FLAGS)
  model_fn = functools.partial(get_classificationxlnet_model, model_config,
                               run_config, FLAGS.n_class, FLAGS.summary_type)
  input_meta_data = {}
  input_meta_data["d_model"] = FLAGS.d_model
  input_meta_data["mem_len"] = FLAGS.mem_len
  input_meta_data["batch_size_per_core"] = int(FLAGS.train_batch_size /
                                               strategy.num_replicas_in_sync)
  input_meta_data["n_layer"] = FLAGS.n_layer
  input_meta_data["lr_layer_decay_rate"] = FLAGS.lr_layer_decay_rate
  input_meta_data["n_class"] = FLAGS.n_class

  training_utils.train(
      strategy=strategy,
      model_fn=model_fn,
      input_meta_data=input_meta_data,
      eval_fn=eval_fn,
      metric_fn=get_metric_fn,
      train_input_fn=train_input_fn,
      init_checkpoint=FLAGS.init_checkpoint,
      init_from_transformerxl=FLAGS.init_from_transformerxl,
      total_training_steps=total_training_steps,
      steps_per_loop=steps_per_loop,
      optimizer=optimizer,
      learning_rate_fn=learning_rate_fn,
      model_dir=FLAGS.model_dir,
      save_steps=FLAGS.save_steps)


if __name__ == "__main__":
  app.run(main)
