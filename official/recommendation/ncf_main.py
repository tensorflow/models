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
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import heapq
import math
import multiprocessing
import os
import signal
import typing

# pylint: disable=g-bad-import-order
import numpy as np
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from tensorflow.contrib.compiler import xla
from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation import model_runner
from official.recommendation import neumf_model
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers


FLAGS = flags.FLAGS


def construct_estimator(num_gpus, model_dir, iterations, params, batch_size,
                        eval_batch_size):
  """Construct either an Estimator or TPUEstimator for NCF.

  Args:
    num_gpus: The number of gpus (Used to select distribution strategy)
    model_dir: The model directory for the estimator
    iterations:  Estimator iterations
    params: The params dict for the estimator
    batch_size: The mini-batch size for training.
    eval_batch_size: The batch size used during evaluation.

  Returns:
    An Estimator or TPUEstimator.
  """

  if params["use_tpu"]:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=params["tpu"],
        zone=params["tpu_zone"],
        project=params["tpu_gcp_project"],
    )
    tf.logging.info("Issuing reset command to TPU to ensure a clean state.")
    tf.Session.reset(tpu_cluster_resolver.get_master())

    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=iterations,
        num_shards=8)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=model_dir,
        save_checkpoints_secs=600,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False),
        tpu_config=tpu_config)

    tpu_params = {k: v for k, v in params.items() if k != "batch_size"}

    train_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=neumf_model.neumf_model_fn,
        use_tpu=True,
        train_batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        params=tpu_params,
        config=run_config)

    eval_estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=neumf_model.neumf_model_fn,
        use_tpu=True,
        train_batch_size=1,
        eval_batch_size=eval_batch_size,
        params=tpu_params,
        config=run_config)

    return train_estimator, eval_estimator

  distribution = distribution_utils.get_distribution_strategy(num_gpus=num_gpus)
  run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                      eval_distribute=distribution)
  params["eval_batch_size"] = eval_batch_size
  model_fn = neumf_model.neumf_model_fn
  if params["use_xla_for_gpu"]:
    tf.logging.info("Using XLA for GPU for training and evaluation.")
    model_fn = xla.estimator_model_fn(model_fn)
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                     config=run_config, params=params)
  return estimator, estimator


def main(_):
  with logger.benchmark_context(FLAGS), \
       mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
    mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
    run_ncf(FLAGS)
    mlperf_helper.stitch_ncf()


def run_ncf(_):
  """Run NCF training and eval loop."""
  if FLAGS.download_if_missing and not FLAGS.use_synthetic_data:
    movielens.download(FLAGS.dataset, FLAGS.data_dir)

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)

  num_gpus = flags_core.get_num_gpus(FLAGS)
  batch_size = distribution_utils.per_device_batch_size(
      int(FLAGS.batch_size), num_gpus)
  total_training_cycle = FLAGS.train_epochs // FLAGS.epochs_between_evals

  eval_per_user = rconst.NUM_EVAL_NEGATIVES + 1
  eval_batch_size = int(FLAGS.eval_batch_size or
                        max([FLAGS.batch_size, eval_per_user]))
  if eval_batch_size % eval_per_user:
    eval_batch_size = eval_batch_size // eval_per_user * eval_per_user
    tf.logging.warning(
        "eval examples per user does not evenly divide eval_batch_size. "
        "Overriding to {}".format(eval_batch_size))

  if FLAGS.use_synthetic_data:
    ncf_dataset = None
    cleanup_fn = lambda: None
    num_users, num_items = data_preprocessing.DATASET_TO_NUM_USERS_AND_ITEMS[
        FLAGS.dataset]
    num_train_steps = data_preprocessing.SYNTHETIC_BATCHES_PER_EPOCH
    num_eval_steps = data_preprocessing.SYNTHETIC_BATCHES_PER_EPOCH
  else:
    ncf_dataset, cleanup_fn = data_preprocessing.instantiate_pipeline(
        dataset=FLAGS.dataset, data_dir=FLAGS.data_dir,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_neg=FLAGS.num_neg,
        epochs_per_cycle=FLAGS.epochs_between_evals,
        num_cycles=total_training_cycle,
        match_mlperf=FLAGS.ml_perf,
        deterministic=FLAGS.seed is not None,
        use_subprocess=FLAGS.use_subprocess,
        cache_id=FLAGS.cache_id)
    num_users = ncf_dataset.num_users
    num_items = ncf_dataset.num_items
    num_train_steps = int(np.ceil(
        FLAGS.epochs_between_evals * ncf_dataset.num_train_positives *
        (1 + FLAGS.num_neg) / FLAGS.batch_size))
    num_eval_steps = int(np.ceil((1 + rconst.NUM_EVAL_NEGATIVES) *
                                 ncf_dataset.num_users / eval_batch_size))

  model_helpers.apply_clean(flags.FLAGS)

  params = {
      "use_seed": FLAGS.seed is not None,
      "hash_pipeline": FLAGS.hash_pipeline,
      "batch_size": batch_size,
      "eval_batch_size": eval_batch_size,
      "learning_rate": FLAGS.learning_rate,
      "num_users": num_users,
      "num_items": num_items,
      "mf_dim": FLAGS.num_factors,
      "model_layers": [int(layer) for layer in FLAGS.layers],
      "mf_regularization": FLAGS.mf_regularization,
      "mlp_reg_layers": [float(reg) for reg in FLAGS.mlp_regularization],
      "num_neg": FLAGS.num_neg,
      "use_tpu": FLAGS.tpu is not None,
      "tpu": FLAGS.tpu,
      "tpu_zone": FLAGS.tpu_zone,
      "tpu_gcp_project": FLAGS.tpu_gcp_project,
      "beta1": FLAGS.beta1,
      "beta2": FLAGS.beta2,
      "epsilon": FLAGS.epsilon,
      "match_mlperf": FLAGS.ml_perf,
      "use_xla_for_gpu": FLAGS.use_xla_for_gpu,
      "use_estimator": FLAGS.use_estimator,
  }
  if FLAGS.use_estimator:
    train_estimator, eval_estimator = construct_estimator(
        num_gpus=num_gpus, model_dir=FLAGS.model_dir,
        iterations=num_train_steps, params=params,
        batch_size=flags.FLAGS.batch_size, eval_batch_size=eval_batch_size)
  else:
    runner = model_runner.NcfModelRunner(ncf_dataset, params, num_train_steps,
                                         num_eval_steps, FLAGS.use_while_loop)

  # Create hooks that log information about the training and metric values
  train_hooks = hooks_helper.get_train_hooks(
      FLAGS.hooks,
      model_dir=FLAGS.model_dir,
      batch_size=FLAGS.batch_size,  # for ExamplesPerSecondHook
      tensors_to_log={"cross_entropy": "cross_entropy"}
  )
  run_params = {
      "batch_size": FLAGS.batch_size,
      "eval_batch_size": eval_batch_size,
      "number_factors": FLAGS.num_factors,
      "hr_threshold": FLAGS.hr_threshold,
      "train_epochs": FLAGS.train_epochs,
  }
  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info(
      model_name="recommendation",
      dataset_name=FLAGS.dataset,
      run_params=run_params,
      test_id=FLAGS.benchmark_test_id)


  eval_input_fn = None
  target_reached = False
  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.TRAIN_LOOP)
  for cycle_index in range(total_training_cycle):
    assert FLAGS.epochs_between_evals == 1 or not mlperf_helper.LOGGER.enabled
    tf.logging.info("Starting a training cycle: {}/{}".format(
        cycle_index + 1, total_training_cycle))

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.TRAIN_EPOCH,
                            value=cycle_index)

    # Train the model
    if FLAGS.use_estimator:
      train_input_fn, train_record_dir, batch_count = \
        data_preprocessing.make_input_fn(
            ncf_dataset=ncf_dataset, is_training=True)

      if batch_count != num_train_steps:
        raise ValueError(
            "Step counts do not match. ({} vs. {}) The async process is "
            "producing incorrect shards.".format(batch_count, num_train_steps))

      train_estimator.train(input_fn=train_input_fn, hooks=train_hooks,
                            steps=num_train_steps)
      if train_record_dir:
        tf.gfile.DeleteRecursively(train_record_dir)

      tf.logging.info("Beginning evaluation.")
      if eval_input_fn is None:
        eval_input_fn, _, eval_batch_count = data_preprocessing.make_input_fn(
            ncf_dataset=ncf_dataset, is_training=False)

        if eval_batch_count != num_eval_steps:
          raise ValueError(
              "Step counts do not match. ({} vs. {}) The async process is "
              "producing incorrect shards.".format(
                  eval_batch_count, num_eval_steps))

      mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_START,
                              value=cycle_index)
      eval_results = eval_estimator.evaluate(eval_input_fn,
                                             steps=num_eval_steps)
      tf.logging.info("Evaluation complete.")
    else:
      runner.train()
      tf.logging.info("Beginning evaluation.")
      mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_START,
                              value=cycle_index)
      eval_results = runner.eval()
      tf.logging.info("Evaluation complete.")
    hr = float(eval_results[rconst.HR_KEY])
    ndcg = float(eval_results[rconst.NDCG_KEY])

    mlperf_helper.ncf_print(
        key=mlperf_helper.TAGS.EVAL_TARGET,
        value={"epoch": cycle_index, "value": FLAGS.hr_threshold})
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_ACCURACY,
                            value={"epoch": cycle_index, "value": hr})
    mlperf_helper.ncf_print(
        key=mlperf_helper.TAGS.EVAL_HP_NUM_NEG,
        value={"epoch": cycle_index, "value": rconst.NUM_EVAL_NEGATIVES})

    # Logged by the async process during record creation.
    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_HP_NUM_USERS,
                            deferred=True)

    mlperf_helper.ncf_print(key=mlperf_helper.TAGS.EVAL_STOP, value=cycle_index)

    # Benchmark the evaluation results
    benchmark_logger.log_evaluation_result(eval_results)
    # Log the HR and NDCG results.
    tf.logging.info(
        "Iteration {}: HR = {:.4f}, NDCG = {:.4f}".format(
            cycle_index + 1, hr, ndcg))

    # If some evaluation threshold is met
    if model_helpers.past_stop_threshold(FLAGS.hr_threshold, hr):
      target_reached = True
      break

  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.RUN_STOP,
                          value={"success": target_reached})
  cleanup_fn()  # Cleanup data construction artifacts and subprocess.

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()

  mlperf_helper.ncf_print(key=mlperf_helper.TAGS.RUN_FINAL)


def define_ncf_flags():
  """Add flags for running ncf_main."""
  # Add common flags
  flags_core.define_base(export_dir=False)
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=False,
      all_reduce_alg=False
  )
  flags_core.define_device(tpu=True)
  flags_core.define_benchmark()

  flags.adopt_module_key_flags(flags_core)

  flags_core.set_defaults(
      model_dir="/tmp/ncf/",
      data_dir="/tmp/movielens-data/",
      train_epochs=2,
      batch_size=256,
      hooks="ProfilerHook",
      tpu=None
  )

  # Add ncf-specific flags
  flags.DEFINE_enum(
      name="dataset", default="ml-1m",
      enum_values=["ml-1m", "ml-20m"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Dataset to be trained and evaluated."))

  flags.DEFINE_boolean(
      name="download_if_missing", default=True, help=flags_core.help_wrap(
          "Download data to data_dir if it is not already present."))

  flags.DEFINE_string(
      name="eval_batch_size", default=None, help=flags_core.help_wrap(
          "The batch size used for evaluation. This should generally be larger"
          "than the training batch size as the lack of back propagation during"
          "evaluation can allow for larger batch sizes to fit in memory. If not"
          "specified, the training batch size (--batch_size) will be used."))

  flags.DEFINE_integer(
      name="num_factors", default=8,
      help=flags_core.help_wrap("The Embedding size of MF model."))

  # Set the default as a list of strings to be consistent with input arguments
  flags.DEFINE_list(
      name="layers", default=["64", "32", "16", "8"],
      help=flags_core.help_wrap(
          "The sizes of hidden layers for MLP. Example "
          "to specify different sizes of MLP layers: --layers=32,16,8,4"))

  flags.DEFINE_float(
      name="mf_regularization", default=0.,
      help=flags_core.help_wrap(
          "The regularization factor for MF embeddings. The factor is used by "
          "regularizer which allows to apply penalties on layer parameters or "
          "layer activity during optimization."))

  flags.DEFINE_list(
      name="mlp_regularization", default=["0.", "0.", "0.", "0."],
      help=flags_core.help_wrap(
          "The regularization factor for each MLP layer. See mf_regularization "
          "help for more info about regularization factor."))

  flags.DEFINE_integer(
      name="num_neg", default=4,
      help=flags_core.help_wrap(
          "The Number of negative instances to pair with a positive instance."))

  flags.DEFINE_float(
      name="learning_rate", default=0.001,
      help=flags_core.help_wrap("The learning rate."))

  flags.DEFINE_float(
      name="beta1", default=0.9,
      help=flags_core.help_wrap("beta1 hyperparameter for the Adam optimizer."))

  flags.DEFINE_float(
      name="beta2", default=0.999,
      help=flags_core.help_wrap("beta2 hyperparameter for the Adam optimizer."))

  flags.DEFINE_float(
      name="epsilon", default=1e-8,
      help=flags_core.help_wrap("epsilon hyperparameter for the Adam "
                                "optimizer."))

  flags.DEFINE_float(
      name="hr_threshold", default=None,
      help=flags_core.help_wrap(
          "If passed, training will stop when the evaluation metric HR is "
          "greater than or equal to hr_threshold. For dataset ml-1m, the "
          "desired hr_threshold is 0.68 which is the result from the paper; "
          "For dataset ml-20m, the threshold can be set as 0.95 which is "
          "achieved by MLPerf implementation."))

  flags.DEFINE_bool(
      name="ml_perf", default=False,
      help=flags_core.help_wrap(
          "If set, changes the behavior of the model slightly to match the "
          "MLPerf reference implementations here: \n"
          "https://github.com/mlperf/reference/tree/master/recommendation/"
          "pytorch\n"
          "The two changes are:\n"
          "1. When computing the HR and NDCG during evaluation, remove "
          "duplicate user-item pairs before the computation. This results in "
          "better HRs and NDCGs.\n"
          "2. Use a different soring algorithm when sorting the input data, "
          "which performs better due to the fact the sorting algorithms are "
          "not stable."))

  flags.DEFINE_bool(
      name="output_ml_perf_compliance_logging", default=False,
      help=flags_core.help_wrap(
          "If set, output the MLPerf compliance logging. This is only useful "
          "if one is running the model for MLPerf. See "
          "https://github.com/mlperf/policies/blob/master/training_rules.adoc"
          "#submission-compliance-logs for details. This uses sudo and so may "
          "ask for your password, as root access is needed to clear the system "
          "caches, which is required for MLPerf compliance."
      )
  )

  flags.DEFINE_integer(
      name="seed", default=None, help=flags_core.help_wrap(
          "This value will be used to seed both NumPy and TensorFlow."))

  flags.DEFINE_bool(
      name="hash_pipeline", default=False, help=flags_core.help_wrap(
          "This flag will perform a separate run of the pipeline and hash "
          "batches as they are produced. \nNOTE: this will significantly slow "
          "training. However it is useful to confirm that a random seed is "
          "does indeed make the data pipeline deterministic."))

  @flags.validator("eval_batch_size", "eval_batch_size must be at least {}"
                   .format(rconst.NUM_EVAL_NEGATIVES + 1))
  def eval_size_check(eval_batch_size):
    return (eval_batch_size is None or
            int(eval_batch_size) > rconst.NUM_EVAL_NEGATIVES)

  flags.DEFINE_bool(
      name="use_subprocess", default=True, help=flags_core.help_wrap(
          "By default, ncf_main.py starts async data generation process as a "
          "subprocess. If set to False, ncf_main.py will assume the async data "
          "generation process has already been started by the user."))

  flags.DEFINE_integer(name="cache_id", default=None, help=flags_core.help_wrap(
      "Use a specified cache_id rather than using a timestamp. This is only "
      "needed to synchronize across multiple workers. Generally this flag will "
      "not need to be set."
  ))

  flags.DEFINE_bool(
      name="use_xla_for_gpu", default=False, help=flags_core.help_wrap(
          "If True, use XLA for the model function. Only works when using a "
          "GPU. On TPUs, XLA is always used"))

  xla_message = "--use_xla_for_gpu is incompatible with --tpu"
  @flags.multi_flags_validator(["use_xla_for_gpu", "tpu"], message=xla_message)
  def xla_validator(flag_dict):
    return not flag_dict["use_xla_for_gpu"] or not flag_dict["tpu"]

  flags.DEFINE_bool(
      name="use_estimator", default=True, help=flags_core.help_wrap(
          "If True, use Estimator to train. Setting to False is slightly "
          "faster, but when False, the following are currently unsupported:\n"
          "  * Using TPUs\n"
          "  * Using more than 1 GPU\n"
          "  * Reloading from checkpoints\n"
          "  * Any hooks specified with --hooks\n"))

  flags.DEFINE_bool(
      name="use_while_loop", default=None, help=flags_core.help_wrap(
          "If set, run an entire epoch in a session.run() call using a "
          "TensorFlow while loop. This can improve performance, but will not "
          "print out losses throughout the epoch. Requires "
          "--use_estimator=false"
      ))

  xla_message = "--use_while_loop requires --use_estimator=false"
  @flags.multi_flags_validator(["use_while_loop", "use_estimator"],
                               message=xla_message)
  def while_loop_validator(flag_dict):
    return (not flag_dict["use_while_loop"] or
            not flag_dict["use_estimator"])


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  define_ncf_flags()
  absl_app.run(main)
