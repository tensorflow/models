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
"""Common functionalities used by both Keras and Estimator implementations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# pylint: disable=g-bad-import-order
import numpy as np
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.recommendation import constants as rconst
from official.recommendation import data_pipeline
from official.recommendation import data_preprocessing
from official.recommendation import movielens
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils

FLAGS = flags.FLAGS


def get_inputs(params):
  """Returns some parameters used by the model."""
  if FLAGS.download_if_missing and not FLAGS.use_synthetic_data:
    movielens.download(FLAGS.dataset, FLAGS.data_dir)

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)

  if FLAGS.use_synthetic_data:
    producer = data_pipeline.DummyConstructor()
    num_users, num_items = data_preprocessing.DATASET_TO_NUM_USERS_AND_ITEMS[
        FLAGS.dataset]
    num_train_steps = rconst.SYNTHETIC_BATCHES_PER_EPOCH
    num_eval_steps = rconst.SYNTHETIC_BATCHES_PER_EPOCH
  else:
    num_users, num_items, producer = data_preprocessing.instantiate_pipeline(
        dataset=FLAGS.dataset, data_dir=FLAGS.data_dir, params=params,
        constructor_type=FLAGS.constructor_type,
        deterministic=FLAGS.seed is not None)
    num_train_steps = producer.train_batches_per_epoch
    num_eval_steps = producer.eval_batches_per_epoch

  return num_users, num_items, num_train_steps, num_eval_steps, producer


def parse_flags(flags_obj):
  """Convenience function to turn flags into params."""
  num_gpus = flags_core.get_num_gpus(flags_obj)

  batch_size = flags_obj.batch_size
  eval_batch_size = flags_obj.eval_batch_size or flags_obj.batch_size

  return {
      "train_epochs": flags_obj.train_epochs,
      "batches_per_step": 1,
      "use_seed": flags_obj.seed is not None,
      "batch_size": batch_size,
      "eval_batch_size": eval_batch_size,
      "learning_rate": flags_obj.learning_rate,
      "mf_dim": flags_obj.num_factors,
      "model_layers": [int(layer) for layer in flags_obj.layers],
      "mf_regularization": flags_obj.mf_regularization,
      "mlp_reg_layers": [float(reg) for reg in flags_obj.mlp_regularization],
      "num_neg": flags_obj.num_neg,
      "distribution_strategy": flags_obj.distribution_strategy,
      "num_gpus": num_gpus,
      "use_tpu": flags_obj.tpu is not None,
      "tpu": flags_obj.tpu,
      "tpu_zone": flags_obj.tpu_zone,
      "tpu_gcp_project": flags_obj.tpu_gcp_project,
      "beta1": flags_obj.beta1,
      "beta2": flags_obj.beta2,
      "epsilon": flags_obj.epsilon,
      "match_mlperf": flags_obj.ml_perf,
      "use_xla_for_gpu": flags_obj.use_xla_for_gpu,
      "epochs_between_evals": FLAGS.epochs_between_evals,
      "keras_use_ctl": flags_obj.keras_use_ctl,
      "hr_threshold": flags_obj.hr_threshold,
      "stream_files": flags_obj.tpu is not None,
      "train_dataset_path": flags_obj.train_dataset_path,
      "eval_dataset_path": flags_obj.eval_dataset_path,
      "input_meta_data_path": flags_obj.input_meta_data_path,
  }


def get_v1_distribution_strategy(params):
  """Returns the distribution strategy to use."""
  if params["use_tpu"]:
    # Some of the networking libraries are quite chatty.
    for name in ["googleapiclient.discovery", "googleapiclient.discovery_cache",
                 "oauth2client.transport"]:
      logging.getLogger(name).setLevel(logging.ERROR)

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=params["tpu"],
        zone=params["tpu_zone"],
        project=params["tpu_gcp_project"],
        coordinator_name="coordinator"
    )

    logging.info("Issuing reset command to TPU to ensure a clean state.")
    tf.Session.reset(tpu_cluster_resolver.get_master())

    # Estimator looks at the master it connects to for MonitoredTrainingSession
    # by reading the `TF_CONFIG` environment variable, and the coordinator
    # is used by StreamingFilesDataset.
    tf_config_env = {
        "session_master": tpu_cluster_resolver.get_master(),
        "eval_session_master": tpu_cluster_resolver.get_master(),
        "coordinator": tpu_cluster_resolver.cluster_spec()
                       .as_dict()["coordinator"]
    }
    os.environ["TF_CONFIG"] = json.dumps(tf_config_env)

    distribution = tf.distribute.experimental.TPUStrategy(
        tpu_cluster_resolver, steps_per_run=100)

  else:
    distribution = distribution_utils.get_distribution_strategy(
        num_gpus=params["num_gpus"])

  return distribution


def define_ncf_flags():
  """Add flags for running ncf_main."""
  # Add common flags
  flags_core.define_base(clean=True, train_epochs=True,
                         epochs_between_evals=True, export_dir=False,
                         run_eagerly=True, stop_threshold=True, num_gpu=True,
                         hooks=True, distribution_strategy=True)
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=True,
      all_reduce_alg=False,
      loss_scale=True,
      dynamic_loss_scale=True,
      enable_xla=True,
      force_v2_in_keras_compile=True
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

  flags.DEFINE_integer(
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
      name="hr_threshold", default=1.0,
      help=flags_core.help_wrap(
          "If passed, training will stop when the evaluation metric HR is "
          "greater than or equal to hr_threshold. For dataset ml-1m, the "
          "desired hr_threshold is 0.68 which is the result from the paper; "
          "For dataset ml-20m, the threshold can be set as 0.95 which is "
          "achieved by MLPerf implementation."))

  flags.DEFINE_enum(
      name="constructor_type", default="bisection",
      enum_values=["bisection", "materialized"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Strategy to use for generating false negatives. materialized has a"
          "precompute that scales badly, but a faster per-epoch construction"
          "time and can be faster on very large systems."))

  flags.DEFINE_string(
      name="train_dataset_path",
      default=None,
      help=flags_core.help_wrap("Path to training data."))

  flags.DEFINE_string(
      name="eval_dataset_path",
      default=None,
      help=flags_core.help_wrap("Path to evaluation data."))

  flags.DEFINE_string(
      name="input_meta_data_path",
      default=None,
      help=flags_core.help_wrap("Path to input meta data file."))

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

  @flags.validator("eval_batch_size", "eval_batch_size must be at least {}"
                   .format(rconst.NUM_EVAL_NEGATIVES + 1))
  def eval_size_check(eval_batch_size):
    return (eval_batch_size is None or
            int(eval_batch_size) > rconst.NUM_EVAL_NEGATIVES)

  flags.DEFINE_bool(
      name="use_xla_for_gpu", default=False, help=flags_core.help_wrap(
          "If True, use XLA for the model function. Only works when using a "
          "GPU. On TPUs, XLA is always used"))

  xla_message = "--use_xla_for_gpu is incompatible with --tpu"
  @flags.multi_flags_validator(["use_xla_for_gpu", "tpu"], message=xla_message)
  def xla_validator(flag_dict):
    return not flag_dict["use_xla_for_gpu"] or not flag_dict["tpu"]

  flags.DEFINE_bool(
      name="early_stopping",
      default=False,
      help=flags_core.help_wrap(
          "If True, we stop the training when it reaches hr_threshold"))

  flags.DEFINE_bool(
      name="keras_use_ctl",
      default=False,
      help=flags_core.help_wrap(
          "If True, we use a custom training loop for keras."))


def convert_to_softmax_logits(logits):
  """Convert the logits returned by the base model to softmax logits.

  Args:
    logits: used to create softmax.

  Returns:
    Softmax with the first column of zeros is equivalent to sigmoid.
  """
  softmax_logits = tf.concat([logits * 0, logits], axis=1)
  return softmax_logits
