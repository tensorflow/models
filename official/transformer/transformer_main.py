# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# pylint: disable=g-bad-import-order
from six.moves import xrange  # pylint: disable=redefined-builtin
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.transformer import compute_bleu
from official.transformer import translate
from official.transformer.data_download import VOCAB_FILE
from official.transformer.model import model_params
from official.transformer.model import transformer
from official.transformer.utils import dataset
from official.transformer.utils import metrics
from official.transformer.utils import tokenizer
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers


PARAMS_MAP = {
    "base": model_params.TransformerBaseParams,
    "big": model_params.TransformerBigParams,
}
DEFAULT_TRAIN_EPOCHS = 10
BLEU_DIR = "bleu"
INF = int(1e9)

# Dictionary containing tensors that are logged by the logging hooks. Each item
# maps a string to the tensor name.
TENSORS_TO_LOG = {
    "learning_rate": "model/get_train_op/learning_rate/learning_rate",
    "cross_entropy_loss": "model/cross_entropy"}


def model_fn(features, labels, mode, params):
  """Defines how to train, evaluate and predict from the transformer model."""
  with tf.variable_scope("model"):
    inputs, targets = features, labels

    # Create model and get output logits.
    model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

    output = model(inputs, targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=output)

    logits = output

    # Calculate model loss.
    # xentropy contains the cross entropy loss of every nonpadding token in the
    # targets.
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, targets, params.label_smoothing, params.vocab_size)
    # Compute the weighted mean of the cross entropy losses
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    # Save loss as named tensor that will be logged with the logging hook.
    tf.identity(loss, "cross_entropy")

    if mode == tf.estimator.ModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, predictions={"predictions": logits},
          eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op = get_train_op(loss, params)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  with tf.name_scope("learning_rate"):
    warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())

    learning_rate *= (hidden_size ** -0.5)
    # Apply linear warmup
    learning_rate *= tf.minimum(1.0, step / warmup_steps)
    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    # Create a named tensor that will be logged using the logging hook.
    # The full name includes variable and names scope. In this case, the name
    # is model/get_train_op/learning_rate/learning_rate
    tf.identity(learning_rate, "learning_rate")
    # Save learning rate value to TensorBoard summary.
    tf.summary.scalar("learning_rate", learning_rate)

    return learning_rate


def get_train_op(loss, params):
  """Generate training operation that updates variables based on loss."""
  with tf.variable_scope("get_train_op"):
    learning_rate = get_learning_rate(
        params.learning_rate, params.hidden_size,
        params.learning_rate_warmup_steps)

    # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
    # than the TF core Adam optimizer.
    optimizer = tf.contrib.opt.LazyAdamOptimizer(
        learning_rate,
        beta1=params.optimizer_adam_beta1,
        beta2=params.optimizer_adam_beta2,
        epsilon=params.optimizer_adam_epsilon)

    # Calculate and apply gradients using LazyAdamOptimizer.
    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    train_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")

    # Save gradient norm to Tensorboard
    tf.summary.scalar("global_norm/gradient_norm",
                      tf.global_norm(list(zip(*gradients))[0]))

    return train_op


def translate_and_compute_bleu(estimator, subtokenizer, bleu_source, bleu_ref):
  """Translate file and report the cased and uncased bleu scores."""
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      estimator, subtokenizer, bleu_source, output_file=tmp_filename,
      print_all_translations=False)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def get_global_step(estimator):
  """Return estimator's last checkpoint."""
  return int(estimator.latest_checkpoint().split("-")[-1])


def evaluate_and_log_bleu(estimator, bleu_source, bleu_ref, vocab_file_path):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file_path)

  uncased_score, cased_score = translate_and_compute_bleu(
      estimator, subtokenizer, bleu_source, bleu_ref)

  tf.logging.info("Bleu score (uncased):", uncased_score)
  tf.logging.info("Bleu score (cased):", cased_score)
  return uncased_score, cased_score


def train_schedule(
    estimator, train_eval_iterations, single_iteration_train_steps=None,
    single_iteration_train_epochs=None, train_hooks=None, benchmark_logger=None,
    bleu_source=None, bleu_ref=None, bleu_threshold=None, vocab_file_path=None):
  """Train and evaluate model, and optionally compute model's BLEU score.

  **Step vs. Epoch vs. Iteration**

  Steps and epochs are canonical terms used in TensorFlow and general machine
  learning. They are used to describe running a single process (train/eval):
    - Step refers to running the process through a single or batch of examples.
    - Epoch refers to running the process through an entire dataset.

  E.g. training a dataset with 100 examples. The dataset is
  divided into 20 batches with 5 examples per batch. A single training step
  trains the model on one batch. After 20 training steps, the model will have
  trained on every batch in the dataset, or, in other words, one epoch.

  Meanwhile, iteration is used in this implementation to describe running
  multiple processes (training and eval).
    - A single iteration:
      1. trains the model for a specific number of steps or epochs.
      2. evaluates the model.
      3. (if source and ref files are provided) compute BLEU score.

  This function runs through multiple train+eval+bleu iterations.

  Args:
    estimator: tf.Estimator containing model to train.
    train_eval_iterations: Number of times to repeat the train+eval iteration.
    single_iteration_train_steps: Number of steps to train in one iteration.
    single_iteration_train_epochs: Number of epochs to train in one iteration.
    train_hooks: List of hooks to pass to the estimator during training.
    benchmark_logger: a BenchmarkLogger object that logs evaluation data
    bleu_source: File containing text to be translated for BLEU calculation.
    bleu_ref: File containing reference translations for BLEU calculation.
    bleu_threshold: minimum BLEU score before training is stopped.
    vocab_file_path: Path to vocabulary file used to subtokenize bleu_source.

  Raises:
    ValueError: if both or none of single_iteration_train_steps and
      single_iteration_train_epochs were defined.
  """
  # Ensure that exactly one of single_iteration_train_steps and
  # single_iteration_train_epochs is defined.
  if single_iteration_train_steps is None:
    if single_iteration_train_epochs is None:
      raise ValueError(
          "Exactly one of single_iteration_train_steps or "
          "single_iteration_train_epochs must be defined. Both were none.")
  else:
    if single_iteration_train_epochs is not None:
      raise ValueError(
          "Exactly one of single_iteration_train_steps or "
          "single_iteration_train_epochs must be defined. Both were defined.")

  evaluate_bleu = bleu_source is not None and bleu_ref is not None

  # Print details of training schedule.
  tf.logging.info("Training schedule:")
  if single_iteration_train_epochs is not None:
    tf.logging.info("\t1. Train for %d epochs." % single_iteration_train_epochs)
  else:
    tf.logging.info("\t1. Train for %d steps." % single_iteration_train_steps)
  tf.logging.info("\t2. Evaluate model.")
  if evaluate_bleu:
    tf.logging.info("\t3. Compute BLEU score.")
    if bleu_threshold is not None:
      tf.logging.info("Repeat above steps until the BLEU score reaches %f" %
                      bleu_threshold)
  if not evaluate_bleu or bleu_threshold is None:
    tf.logging.info("Repeat above steps %d times." % train_eval_iterations)

  if evaluate_bleu:
    # Create summary writer to log bleu score (values can be displayed in
    # Tensorboard).
    bleu_writer = tf.summary.FileWriter(
        os.path.join(estimator.model_dir, BLEU_DIR))
    if bleu_threshold is not None:
      # Change loop stopping condition if bleu_threshold is defined.
      train_eval_iterations = INF

  # Loop training/evaluation/bleu cycles
  for i in xrange(train_eval_iterations):
    tf.logging.info("Starting iteration %d" % (i + 1))

    # Train the model for single_iteration_train_steps or until the input fn
    # runs out of examples (if single_iteration_train_steps is None).
    estimator.train(
        dataset.train_input_fn, steps=single_iteration_train_steps,
        hooks=train_hooks)

    eval_results = estimator.evaluate(dataset.eval_input_fn)
    tf.logging.info("Evaluation results (iter %d/%d):" %
                    (i + 1, train_eval_iterations))
    tf.logging.info(eval_results)
    benchmark_logger.log_evaluation_result(eval_results)

    # The results from estimator.evaluate() are measured on an approximate
    # translation, which utilize the target golden values provided. The actual
    # bleu score must be computed using the estimator.predict() path, which
    # outputs translations that are not based on golden values. The translations
    # are compared to reference file to get the actual bleu score.
    if evaluate_bleu:
      uncased_score, cased_score = evaluate_and_log_bleu(
          estimator, bleu_source, bleu_ref, vocab_file_path)

      # Write actual bleu scores using summary writer and benchmark logger
      global_step = get_global_step(estimator)
      summary = tf.Summary(value=[
          tf.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
          tf.Summary.Value(tag="bleu/cased", simple_value=cased_score),
      ])
      bleu_writer.add_summary(summary, global_step)
      bleu_writer.flush()
      benchmark_logger.log_metric(
          "bleu_uncased", uncased_score, global_step=global_step)
      benchmark_logger.log_metric(
          "bleu_cased", cased_score, global_step=global_step)

      # Stop training if bleu stopping threshold is met.
      if model_helpers.past_stop_threshold(bleu_threshold, uncased_score):
        bleu_writer.close()
        break


def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  # Add common flags (data_dir, model_dir, train_epochs, etc.).
  flags_core.define_base(multi_gpu=False, num_gpu=False, export_dir=False)
  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=False
  )
  flags_core.define_benchmark()

  # Set flags from the flags_core module as "key flags" so they're listed when
  # the '-h' flag is used. Without this line, the flags defined above are
  # only shown in the full `--helpful` help text.
  flags.adopt_module_key_flags(flags_core)

  # Add transformer-specific flags
  flags.DEFINE_enum(
      name="param_set", short_name="mp", default="big",
      enum_values=["base", "big"],
      help=flags_core.help_wrap(
          "Parameter set to use when creating and training the model. The "
          "parameters define the input shape (batch size and max length), "
          "model configuration (size of embedding, # of hidden layers, etc.), "
          "and various other settings. The big parameter set increases the "
          "default batch size, embedding/hidden size, and filter size. For a "
          "complete list of parameters, please see model/model_params.py."))

  # Flags for training with steps (may be used for debugging)
  flags.DEFINE_integer(
      name="train_steps", short_name="ts", default=None,
      help=flags_core.help_wrap("The number of steps used to train."))
  flags.DEFINE_integer(
      name="steps_between_evals", short_name="sbe", default=1000,
      help=flags_core.help_wrap(
          "The Number of training steps to run between evaluations. This is "
          "used if --train_steps is defined."))

  # BLEU score computation
  flags.DEFINE_string(
      name="bleu_source", short_name="bls", default=None,
      help=flags_core.help_wrap(
          "Path to source file containing text translate when calculating the "
          "official BLEU score. --bleu_source, --bleu_ref, and --vocab_file "
          "must be set. Use the flag --stop_threshold to stop the script based "
          "on the uncased BLEU score."))
  flags.DEFINE_string(
      name="bleu_ref", short_name="blr", default=None,
      help=flags_core.help_wrap(
          "Path to source file containing text translate when calculating the "
          "official BLEU score. --bleu_source, --bleu_ref, and --vocab_file "
          "must be set. Use the flag --stop_threshold to stop the script based "
          "on the uncased BLEU score."))
  flags.DEFINE_string(
      name="vocab_file", short_name="vf", default=VOCAB_FILE,
      help=flags_core.help_wrap(
          "Name of vocabulary file containing subtokens for subtokenizing the "
          "bleu_source file. This file is expected to be in the directory "
          "defined by --data_dir."))

  flags_core.set_defaults(data_dir="/tmp/translate_ende",
                          model_dir="/tmp/transformer_model",
                          batch_size=None,
                          train_epochs=None)

  @flags.multi_flags_validator(
      ["train_epochs", "train_steps"],
      message="Both --train_steps and --train_epochs were set. Only one may be "
              "defined.")
  def _check_train_limits(flag_dict):
    return flag_dict["train_epochs"] is None or flag_dict["train_steps"] is None

  @flags.multi_flags_validator(
      ["data_dir", "bleu_source", "bleu_ref", "vocab_file"],
      message="--bleu_source, --bleu_ref, and/or --vocab_file don't exist. "
              "Please ensure that the file paths are correct.")
  def _check_bleu_files(flags_dict):
    """Validate files when bleu_source and bleu_ref are defined."""
    if flags_dict["bleu_source"] is None or flags_dict["bleu_ref"] is None:
      return True
    # Ensure that bleu_source, bleu_ref, and vocab files exist.
    vocab_file_path = os.path.join(
        flags_dict["data_dir"], flags_dict["vocab_file"])
    return all([
        tf.gfile.Exists(flags_dict["bleu_source"]),
        tf.gfile.Exists(flags_dict["bleu_ref"]),
        tf.gfile.Exists(vocab_file_path)])


def run_transformer(flags_obj):
  """Create tf.Estimator to train and evaluate transformer model.

  Args:
    flags_obj: Object containing parsed flag values.
  """
  # Determine training schedule based on flags.
  if flags_obj.train_steps is not None:
    train_eval_iterations = (
        flags_obj.train_steps // flags_obj.steps_between_evals)
    single_iteration_train_steps = flags_obj.steps_between_evals
    single_iteration_train_epochs = None
  else:
    train_epochs = flags_obj.train_epochs or DEFAULT_TRAIN_EPOCHS
    train_eval_iterations = train_epochs // flags_obj.epochs_between_evals
    single_iteration_train_steps = None
    single_iteration_train_epochs = flags_obj.epochs_between_evals

  # Add flag-defined parameters to params object
  params = PARAMS_MAP[flags_obj.param_set]
  params.data_dir = flags_obj.data_dir
  params.num_parallel_calls = flags_obj.num_parallel_calls
  params.epochs_between_evals = flags_obj.epochs_between_evals
  params.repeat_dataset = single_iteration_train_epochs
  params.batch_size = flags_obj.batch_size or params.batch_size

  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      tensors_to_log=TENSORS_TO_LOG,  # used for logging hooks
      batch_size=params.batch_size  # for ExamplesPerSecondHook
  )
  benchmark_logger = logger.config_benchmark_logger(flags_obj)
  benchmark_logger.log_run_info(
      model_name="transformer",
      dataset_name="wmt_translate_ende",
      run_params=params.__dict__)

  # Train and evaluate transformer model
  estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=flags_obj.model_dir, params=params)
  train_schedule(
      estimator=estimator,
      # Training arguments
      train_eval_iterations=train_eval_iterations,
      single_iteration_train_steps=single_iteration_train_steps,
      single_iteration_train_epochs=single_iteration_train_epochs,
      train_hooks=train_hooks,
      benchmark_logger=benchmark_logger,
      # BLEU calculation arguments
      bleu_source=flags_obj.bleu_source,
      bleu_ref=flags_obj.bleu_ref,
      bleu_threshold=flags_obj.stop_threshold,
      vocab_file_path=os.path.join(flags_obj.data_dir, flags_obj.vocab_file))


def main(_):
  run_transformer(flags.FLAGS)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_transformer_flags()
  absl_app.run(main)
