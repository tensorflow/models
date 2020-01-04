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

from official.nlp.transformer import model_params
from official.r1.utils import export
from official.r1.utils import tpu as tpu_util
from official.r1.transformer import translate
from official.r1.transformer import transformer
from official.r1.transformer import dataset
from official.r1.transformer import schedule
from official.transformer import compute_bleu
from official.transformer.utils import metrics
from official.transformer.utils import tokenizer
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

PARAMS_MAP = {
    "tiny": model_params.TINY_PARAMS,
    "base": model_params.BASE_PARAMS,
    "big": model_params.BIG_PARAMS,
}


DEFAULT_TRAIN_EPOCHS = 10
INF = 1000000000  # 1e9
BLEU_DIR = "bleu"

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

    logits = model(inputs, targets)

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
      if params["use_tpu"]:
        raise NotImplementedError("Prediction is not yet supported on TPUs.")
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=logits,
          export_outputs={
              "translate": tf.estimator.export.PredictOutput(logits)
          })

    # Explicitly set the shape of the logits for XLA (TPU). This is needed
    # because the logits are passed back to the host VM CPU for metric
    # evaluation, and the shape of [?, ?, vocab_size] is too vague. However
    # it is known from Transformer that the first two dimensions of logits
    # are the dimensions of targets. Note that the ambiguous shape of logits is
    # not a problem when computing xentropy, because padded_cross_entropy_loss
    # resolves the shape on the TPU.
    logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

    # Calculate model loss.
    # xentropy contains the cross entropy loss of every nonpadding token in the
    # targets.
    xentropy, weights = metrics.padded_cross_entropy_loss(
        logits, targets, params["label_smoothing"], params["vocab_size"])
    loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    # Save loss as named tensor that will be logged with the logging hook.
    tf.identity(loss, "cross_entropy")

    if mode == tf.estimator.ModeKeys.EVAL:
      if params["use_tpu"]:
        # host call functions should only have tensors as arguments.
        # This lambda pre-populates params so that metric_fn is
        # TPUEstimator compliant.
        metric_fn = lambda logits, labels: (
            metrics.get_eval_metrics(logits, labels, params=params))
        eval_metrics = (metric_fn, [logits, labels])
        return tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            predictions={"predictions": logits},
            eval_metrics=eval_metrics)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, predictions={"predictions": logits},
          eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
    else:
      train_op, metric_dict = get_train_op_and_metrics(loss, params)

      # Epochs can be quite long. This gives some intermediate information
      # in TensorBoard.
      metric_dict["minibatch_loss"] = loss
      if params["use_tpu"]:
        return tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            host_call=tpu_util.construct_scalar_host_call(
                metric_dict=metric_dict,
                model_dir=params["model_dir"],
                prefix="training/"))
      record_scalars(metric_dict)
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def record_scalars(metric_dict):
  for key, value in metric_dict.items():
    tf.summary.scalar(name=key, tensor=value)


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

    return learning_rate


def get_train_op_and_metrics(loss, params):
  """Generate training op and metrics to save in TensorBoard."""
  with tf.variable_scope("get_train_op"):
    learning_rate = get_learning_rate(
        learning_rate=params["learning_rate"],
        hidden_size=params["hidden_size"],
        learning_rate_warmup_steps=params["learning_rate_warmup_steps"])

    # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
    # than the TF core Adam optimizer.
    from tensorflow.contrib import opt as contrib_opt  # pylint: disable=g-import-not-at-top
    optimizer = contrib_opt.LazyAdamOptimizer(
        learning_rate,
        beta1=params["optimizer_adam_beta1"],
        beta2=params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    if params["use_tpu"] and params["tpu"] != tpu_util.LOCAL:
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

    # Uses automatic mixed precision FP16 training if on GPU.
    if params["dtype"] == "fp16":
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer)

    # Calculate and apply gradients using LazyAdamOptimizer.
    global_step = tf.train.get_global_step()
    tvars = tf.trainable_variables()
    gradients = optimizer.compute_gradients(
        loss, tvars, colocate_gradients_with_ops=True)
    minimize_op = optimizer.apply_gradients(
        gradients, global_step=global_step, name="train")
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

    train_metrics = {"learning_rate": learning_rate}

    if not params["use_tpu"]:
      # gradient norm is not included as a summary when running on TPU, as
      # it can cause instability between the TPU and the host controller.
      gradient_norm = tf.global_norm(list(zip(*gradients))[0])
      train_metrics["global_norm/gradient_norm"] = gradient_norm

    return train_op, train_metrics


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


def evaluate_and_log_bleu(estimator, bleu_source, bleu_ref, vocab_file):
  """Calculate and record the BLEU score."""
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      estimator, subtokenizer, bleu_source, bleu_ref)

  tf.logging.info("Bleu score (uncased): %f", uncased_score)
  tf.logging.info("Bleu score (cased): %f", cased_score)
  return uncased_score, cased_score


def _validate_file(filepath):
  """Make sure that file exists."""
  if not tf.io.gfile.exists(filepath):
    raise tf.errors.NotFoundError(None, None, "File %s not found." % filepath)


def run_loop(
    estimator, schedule_manager, train_hooks=None, benchmark_logger=None,
    bleu_source=None, bleu_ref=None, bleu_threshold=None, vocab_file=None):
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
    schedule_manager: A schedule.Manager object to guide the run loop.
    train_hooks: List of hooks to pass to the estimator during training.
    benchmark_logger: a BenchmarkLogger object that logs evaluation data
    bleu_source: File containing text to be translated for BLEU calculation.
    bleu_ref: File containing reference translations for BLEU calculation.
    bleu_threshold: minimum BLEU score before training is stopped.
    vocab_file: Path to vocab file that will be used to subtokenize bleu_source.

  Returns:
    Dict of results of the run.  Contains the keys `eval_results`,
    `train_hooks`, `bleu_cased`, and `bleu_uncased`. `train_hooks` is a list the
    instances of hooks used during training.

  Raises:
    ValueError: if both or none of single_iteration_train_steps and
      single_iteration_train_epochs were defined.
    NotFoundError: if the vocab file or bleu files don't exist.
  """
  if bleu_source:
    _validate_file(bleu_source)
  if bleu_ref:
    _validate_file(bleu_ref)
  if vocab_file:
    _validate_file(vocab_file)

  evaluate_bleu = bleu_source is not None and bleu_ref is not None
  if evaluate_bleu and schedule_manager.use_tpu:
    raise ValueError("BLEU score can not be computed when training with a TPU, "
                     "as it requires estimator.predict which is not yet "
                     "supported.")

  # Print details of training schedule.
  tf.logging.info("Training schedule:")
  tf.logging.info(
      "\t1. Train for {}".format(schedule_manager.train_increment_str))
  tf.logging.info("\t2. Evaluate model.")
  if evaluate_bleu:
    tf.logging.info("\t3. Compute BLEU score.")
    if bleu_threshold is not None:
      tf.logging.info("Repeat above steps until the BLEU score reaches %f" %
                      bleu_threshold)
  if not evaluate_bleu or bleu_threshold is None:
    tf.logging.info("Repeat above steps %d times." %
                    schedule_manager.train_eval_iterations)

  if evaluate_bleu:
    # Create summary writer to log bleu score (values can be displayed in
    # Tensorboard).
    bleu_writer = tf.summary.FileWriter(
        os.path.join(estimator.model_dir, BLEU_DIR))
    if bleu_threshold is not None:
      # Change loop stopping condition if bleu_threshold is defined.
      schedule_manager.train_eval_iterations = INF

  # Loop training/evaluation/bleu cycles
  stats = {}
  for i in xrange(schedule_manager.train_eval_iterations):
    tf.logging.info("Starting iteration %d" % (i + 1))

    # Train the model for single_iteration_train_steps or until the input fn
    # runs out of examples (if single_iteration_train_steps is None).
    estimator.train(
        dataset.train_input_fn,
        steps=schedule_manager.single_iteration_train_steps,
        hooks=train_hooks)

    eval_results = None
    eval_results = estimator.evaluate(
        input_fn=dataset.eval_input_fn,
        steps=schedule_manager.single_iteration_eval_steps)

    tf.logging.info("Evaluation results (iter %d/%d):" %
                    (i + 1, schedule_manager.train_eval_iterations))
    tf.logging.info(eval_results)
    benchmark_logger.log_evaluation_result(eval_results)

    # The results from estimator.evaluate() are measured on an approximate
    # translation, which utilize the target golden values provided. The actual
    # bleu score must be computed using the estimator.predict() path, which
    # outputs translations that are not based on golden values. The translations
    # are compared to reference file to get the actual bleu score.
    if evaluate_bleu:
      uncased_score, cased_score = evaluate_and_log_bleu(
          estimator, bleu_source, bleu_ref, vocab_file)

      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score

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

  stats["eval_results"] = eval_results
  stats["train_hooks"] = train_hooks

  return stats


def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  # Add common flags (data_dir, model_dir, train_epochs, etc.).
  flags.DEFINE_integer(
      name="max_length", short_name="ml", default=None,
      help=flags_core.help_wrap("Max length."))

  flags_core.define_base(clean=True, train_epochs=True,
                         epochs_between_evals=True, stop_threshold=True,
                         num_gpu=True, hooks=True, export_dir=True,
                         distribution_strategy=True)
  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=True,
      all_reduce_alg=True
  )
  flags_core.define_benchmark()
  flags_core.define_device(tpu=True)

  # Set flags from the flags_core module as "key flags" so they're listed when
  # the '-h' flag is used. Without this line, the flags defined above are
  # only shown in the full `--helpful` help text.
  flags.adopt_module_key_flags(flags_core)

  # Add transformer-specific flags
  flags.DEFINE_enum(
      name="param_set", short_name="mp", default="big",
      enum_values=PARAMS_MAP.keys(),
      help=flags_core.help_wrap(
          "Parameter set to use when creating and training the model. The "
          "parameters define the input shape (batch size and max length), "
          "model configuration (size of embedding, # of hidden layers, etc.), "
          "and various other settings. The big parameter set increases the "
          "default batch size, embedding/hidden size, and filter size. For a "
          "complete list of parameters, please see model/model_params.py."))

  flags.DEFINE_bool(
      name="static_batch", default=False,
      help=flags_core.help_wrap(
          "Whether the batches in the dataset should have static shapes. In "
          "general, this setting should be False. Dynamic shapes allow the "
          "inputs to be grouped so that the number of padding tokens is "
          "minimized, and helps model training. In cases where the input shape "
          "must be static (e.g. running on TPU), this setting will be ignored "
          "and static batching will always be used."))

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
          "official BLEU score. Both --bleu_source and --bleu_ref must be set. "
          "Use the flag --stop_threshold to stop the script based on the "
          "uncased BLEU score."))
  flags.DEFINE_string(
      name="bleu_ref", short_name="blr", default=None,
      help=flags_core.help_wrap(
          "Path to source file containing text translate when calculating the "
          "official BLEU score. Both --bleu_source and --bleu_ref must be set. "
          "Use the flag --stop_threshold to stop the script based on the "
          "uncased BLEU score."))
  flags.DEFINE_string(
      name="vocab_file", short_name="vf", default=None,
      help=flags_core.help_wrap(
          "Path to subtoken vocabulary file. If data_download.py was used to "
          "download and encode the training data, look in the data_dir to find "
          "the vocab file."))

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
      ["bleu_source", "bleu_ref"],
      message="Both or neither --bleu_source and --bleu_ref must be defined.")
  def _check_bleu_files(flags_dict):
    return (flags_dict["bleu_source"] is None) == (
        flags_dict["bleu_ref"] is None)

  @flags.multi_flags_validator(
      ["bleu_source", "bleu_ref", "vocab_file"],
      message="--vocab_file must be defined if --bleu_source and --bleu_ref "
              "are defined.")
  def _check_bleu_vocab_file(flags_dict):
    if flags_dict["bleu_source"] and flags_dict["bleu_ref"]:
      return flags_dict["vocab_file"] is not None
    return True

  @flags.multi_flags_validator(
      ["export_dir", "vocab_file"],
      message="--vocab_file must be defined if --export_dir is set.")
  def _check_export_vocab_file(flags_dict):
    if flags_dict["export_dir"]:
      return flags_dict["vocab_file"] is not None
    return True

  flags_core.require_cloud_storage(["data_dir", "model_dir", "export_dir"])


def construct_estimator(flags_obj, params, schedule_manager):
  """Construct an estimator from either Estimator or TPUEstimator.

  Args:
    flags_obj: The FLAGS object parsed from command line.
    params: A dict of run specific parameters.
    schedule_manager: A schedule.Manager object containing the run schedule.

  Returns:
    An estimator object to be used for training and eval.
  """
  if not params["use_tpu"]:
    distribution_strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=flags_core.get_num_gpus(flags_obj),
        all_reduce_alg=flags_obj.all_reduce_alg)
    return tf.estimator.Estimator(
        model_fn=model_fn, model_dir=flags_obj.model_dir, params=params,
        config=tf.estimator.RunConfig(train_distribute=distribution_strategy))

  tpu_cluster_resolver = tf.compat.v1.cluster_resolver.TPUClusterResolver(
      tpu=flags_obj.tpu,
      zone=flags_obj.tpu_zone,
      project=flags_obj.tpu_gcp_project
  )

  tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=schedule_manager.single_iteration_train_steps,
      num_shards=flags_obj.num_tpu_shards)

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=flags_obj.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tpu_config)

  return tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=params["use_tpu"] and flags_obj.tpu != tpu_util.LOCAL,
      train_batch_size=schedule_manager.batch_size,
      eval_batch_size=schedule_manager.batch_size,
      params={
          # TPUEstimator needs to populate batch_size itself due to sharding.
          key: value for key, value in params.items() if key != "batch_size"
      },
      config=run_config)


def run_transformer(flags_obj):
  """Create tf.Estimator to train and evaluate transformer model.

  Args:
    flags_obj: Object containing parsed flag values.

  Returns:
    Dict of results of the run.  Contains the keys `eval_results`,
    `train_hooks`, `bleu_cased`, and `bleu_uncased`. `train_hooks` is a list the
    instances of hooks used during training.
  """
  num_gpus = flags_core.get_num_gpus(flags_obj)

  # Add flag-defined parameters to params object
  params = PARAMS_MAP[flags_obj.param_set]
  if num_gpus > 1:
    if flags_obj.param_set == "big":
      params = model_params.BIG_MULTI_GPU_PARAMS
    elif flags_obj.param_set == "base":
      params = model_params.BASE_MULTI_GPU_PARAMS

  params["data_dir"] = flags_obj.data_dir
  params["model_dir"] = flags_obj.model_dir
  params["num_parallel_calls"] = flags_obj.num_parallel_calls

  params["tpu"] = flags_obj.tpu
  params["use_tpu"] = bool(flags_obj.tpu)  # was a tpu specified.
  params["static_batch"] = flags_obj.static_batch or params["use_tpu"]
  params["allow_ffn_pad"] = not params["use_tpu"]

  params["max_length"] = flags_obj.max_length or params["max_length"]

  params["use_synthetic_data"] = flags_obj.use_synthetic_data

  # Set batch size parameter, which depends on the availability of
  # TPU and GPU, and distribution settings.
  params["batch_size"] = (flags_obj.batch_size or (
      params["default_batch_size_tpu"] if params["use_tpu"]
      else params["default_batch_size"]))

  total_batch_size = params["batch_size"]
  if not params["use_tpu"]:
    params["batch_size"] = distribution_utils.per_replica_batch_size(
        params["batch_size"], num_gpus)

  schedule_manager = schedule.Manager(
      train_steps=flags_obj.train_steps,
      steps_between_evals=flags_obj.steps_between_evals,
      train_epochs=flags_obj.train_epochs,
      epochs_between_evals=flags_obj.epochs_between_evals,
      default_train_epochs=DEFAULT_TRAIN_EPOCHS,
      batch_size=params["batch_size"],
      max_length=params["max_length"],
      use_tpu=params["use_tpu"],
      num_tpu_shards=flags_obj.num_tpu_shards
  )

  params["repeat_dataset"] = schedule_manager.repeat_dataset

  model_helpers.apply_clean(flags.FLAGS)

  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=flags_obj.model_dir,
      tensors_to_log=TENSORS_TO_LOG,  # used for logging hooks
      batch_size=total_batch_size,  # for ExamplesPerSecondHook
      use_tpu=params["use_tpu"]  # Not all hooks can run with TPUs
  )
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name="transformer",
      dataset_name="wmt_translate_ende",
      run_params=params,
      test_id=flags_obj.benchmark_test_id)

  # Train and evaluate transformer model
  estimator = construct_estimator(flags_obj, params, schedule_manager)
  stats = run_loop(
      estimator=estimator,
      # Training arguments
      schedule_manager=schedule_manager,
      train_hooks=train_hooks,
      benchmark_logger=benchmark_logger,
      # BLEU calculation arguments
      bleu_source=flags_obj.bleu_source,
      bleu_ref=flags_obj.bleu_ref,
      bleu_threshold=flags_obj.stop_threshold,
      vocab_file=flags_obj.vocab_file)

  if flags_obj.export_dir and not params["use_tpu"]:
    serving_input_fn = export.build_tensor_serving_input_receiver_fn(
        shape=[None], dtype=tf.int64, batch_size=None)
    # Export saved model, and save the vocab file as an extra asset. The vocab
    # file is saved to allow consistent input encoding and output decoding.
    # (See the "Export trained model" section in the README for an example of
    # how to use the vocab file.)
    # Since the model itself does not use the vocab file, this file is saved as
    # an extra asset rather than a core asset.
    estimator.export_savedmodel(
        flags_obj.export_dir, serving_input_fn,
        assets_extra={"vocab.txt": flags_obj.vocab_file},
        strip_default_attrs=True)
  return stats


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_transformer(flags.FLAGS)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_transformer_flags()
  absl_app.run(main)
