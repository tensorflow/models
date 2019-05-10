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
"""Register flags for optimizing performance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

from absl import flags    # pylint: disable=g-bad-import-order
import tensorflow as tf   # pylint: disable=g-bad-import-order

from official.utils.flags._conventions import help_wrap


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
    "fp16": (tf.float16, 128),
    "fp32": (tf.float32, 1),
}


def get_tf_dtype(flags_obj):
  if getattr(flags_obj, 'fp16_implementation', None) == 'graph_rewrite':
    # If the graph_rewrite is used, we build the graph with fp32, and let the
    # graph rewrite change ops to fp16.
    return tf.float32
  return DTYPE_MAP[flags_obj.dtype][0]


def get_loss_scale(flags_obj):
  if flags_obj.loss_scale == "dynamic":
    return flags_obj.loss_scale
  elif flags_obj.loss_scale is not None:
    return float(flags_obj.loss_scale)
  return DTYPE_MAP[flags_obj.dtype][1]


def define_performance(num_parallel_calls=True, inter_op=True, intra_op=True,
                       synthetic_data=True, max_train_steps=True, dtype=True,
                       all_reduce_alg=True, num_packs=True,
                       tf_gpu_thread_mode=False,
                       datasets_num_private_threads=False,
                       datasets_num_parallel_batches=False,
                       dynamic_loss_scale=False, fp16_implementation=False):
  """Register flags for specifying performance tuning arguments.

  Args:
    num_parallel_calls: Create a flag to specify parallelism of data loading.
    inter_op: Create a flag to allow specification of inter op threads.
    intra_op: Create a flag to allow specification of intra op threads.
    synthetic_data: Create a flag to allow the use of synthetic data.
    max_train_steps: Create a flags to allow specification of maximum number
      of training steps
    dtype: Create flags for specifying dtype.
    all_reduce_alg: If set forces a specific algorithm for multi-gpu.
    num_packs: If set provides number of packs for MirroredStrategy's cross
      device ops.
    tf_gpu_thread_mode: gpu_private triggers us of private thread pool.
    datasets_num_private_threads: Number of private threads for datasets.
    datasets_num_parallel_batches: Determines how many batches to process in
    parallel when using map and batch from tf.data.
    dynamic_loss_scale: Allow the "loss_scale" flag to take on the value
      "dynamic". Only valid if `dtype` is True.
    fp16_implementation: Create fp16_implementation flag.

  Returns:
    A list of flags for core.py to marks as key flags.
  """

  key_flags = []
  if num_parallel_calls:
    flags.DEFINE_integer(
        name="num_parallel_calls", short_name="npc",
        default=multiprocessing.cpu_count(),
        help=help_wrap("The number of records that are  processed in parallel "
                       "during input processing. This can be optimized per "
                       "data set but for generally homogeneous data sets, "
                       "should be approximately the number of available CPU "
                       "cores. (default behavior)"))

  if inter_op:
    flags.DEFINE_integer(
        name="inter_op_parallelism_threads", short_name="inter", default=0,
        help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details.")
    )

  if intra_op:
    flags.DEFINE_integer(
        name="intra_op_parallelism_threads", short_name="intra", default=0,
        help=help_wrap("Number of intra_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details."))

  if synthetic_data:
    flags.DEFINE_bool(
        name="use_synthetic_data", short_name="synth", default=False,
        help=help_wrap(
            "If set, use fake data (zeroes) instead of a real dataset. "
            "This mode is useful for performance debugging, as it removes "
            "input processing steps, but will not learn anything."))

  if max_train_steps:
    flags.DEFINE_integer(
        name="max_train_steps", short_name="mts", default=None, help=help_wrap(
            "The model will stop training if the global_step reaches this "
            "value. If not set, training will run until the specified number "
            "of epochs have run as usual. It is generally recommended to set "
            "--train_epochs=1 when using this flag."
        ))

  if dtype:
    flags.DEFINE_enum(
        name="dtype", short_name="dt", default="fp32",
        enum_values=DTYPE_MAP.keys(),
        help=help_wrap("The TensorFlow datatype used for calculations. "
                       "Variables may be cast to a higher precision on a "
                       "case-by-case basis for numerical stability."))

    loss_scale_help_text = (
        "The amount to scale the loss by when the model is run. {}. Before "
        "gradients are computed, the loss is multiplied by the loss scale, "
        "making all gradients loss_scale times larger. To adjust for this, "
        "gradients are divided by the loss scale before being applied to "
        "variables. This is mathematically equivalent to training without "
        "a loss scale, but the loss scale helps avoid some intermediate "
        "gradients from underflowing to zero. If not provided the default "
        "for fp16 is 128 and 1 for all other dtypes.{}"
    )
    if dynamic_loss_scale:
      loss_scale_help_text = loss_scale_help_text.format(
          "This can be an int/float or the string 'dynamic'",
          " The string 'dynamic' can be used to dynamically determine the "
          "optimal loss scale during training, but currently this "
          "significantly slows down performance")
      loss_scale_validation_msg = ("loss_scale should be a positive int/float "
                                   "or the string 'dynamic'.")
    else:
      loss_scale_help_text = loss_scale_help_text.format(
          "This must be an int/float", "")
      loss_scale_validation_msg = "loss_scale should be a positive int/float."
    flags.DEFINE_string(
        name="loss_scale", short_name="ls", default=None,
        help=help_wrap(loss_scale_help_text))

    @flags.validator(flag_name="loss_scale", message=loss_scale_validation_msg)
    def _check_loss_scale(loss_scale):  # pylint: disable=unused-variable
      """Validator to check the loss scale flag is valid"""
      if loss_scale is None:
        return True  # null case is handled in get_loss_scale()

      if loss_scale == "dynamic" and dynamic_loss_scale:
        return True

      try:
        loss_scale = float(loss_scale)
      except ValueError:
        return False

      return loss_scale > 0

    if fp16_implementation:
      # Currently, this flag is only defined for the estimator resnet model.
      flags.DEFINE_enum(
          name="fp16_implementation", default='casting',
          enum_values=('casting', 'graph_rewrite'),
          help=help_wrap(
              "When --dtype=fp16, how fp16 should be implemented. This has no "
              "impact on correctness. 'casting' will cause manual tf.casts to "
              "be inserted in the model. 'graph_rewrite' means "
              "tf.train.experimental.enable_mixed_precision_graph_rewrite will "
              "be used to automatically use fp16 without any manual casts."))

      @flags.multi_flags_validator(['fp16_implementation', 'dtype',
                                    'loss_scale'])
      def _check_fp16_implementation(flags_dict):
        """Validator to check fp16_implementation flag is valid."""
        if (flags_dict['fp16_implementation'] == 'graph_rewrite' and
            flags_dict['dtype'] != 'fp16'):
          raise flags.ValidationError('--fp16_implementation should not be '
                                      'specified unless --dtype=fp16')
        if (flags_dict['fp16_implementation'] != 'graph_rewrite' and
            flags_dict['loss_scale'] == 'dynamic'):
          raise flags.ValidationError('--loss_scale=dynamic is only supported '
                                      'when '
                                      '--fp16_implementation=graph_rewrite')
        return True

  if all_reduce_alg:
    flags.DEFINE_string(
        name="all_reduce_alg", short_name="ara", default=None,
        help=help_wrap("Defines the algorithm to use for performing all-reduce."
                       "When specified with MirroredStrategy for single "
                       "worker, this controls "
                       "tf.contrib.distribute.AllReduceCrossTowerOps.  When "
                       "specified with MultiWorkerMirroredStrategy, this "
                       "controls "
                       "tf.distribute.experimental.CollectiveCommunication; "
                       "valid options are `ring` and `nccl`."))

  if num_packs:
    flags.DEFINE_integer(
        name="num_packs", default=1,
        help=help_wrap("Sets `num_packs` in the cross device ops used in "
                       "MirroredStrategy.  For details, see "
                       "tf.distribute.NcclAllReduce."))

  if tf_gpu_thread_mode:
    flags.DEFINE_string(
        name="tf_gpu_thread_mode", short_name="gt_mode", default=None,
        help=help_wrap(
            "Whether and how the GPU device uses its own threadpool.")
    )

    flags.DEFINE_integer(
        name="per_gpu_thread_count", short_name="pgtc", default=0,
        help=help_wrap(
            "The number of threads to use for GPU. Only valid when "
            "tf_gpu_thread_mode is not global.")
    )

  if datasets_num_private_threads:
    flags.DEFINE_integer(
        name="datasets_num_private_threads",
        default=None,
        help=help_wrap(
            "Number of threads for a private threadpool created for all"
            "datasets computation..")
    )

  if datasets_num_parallel_batches:
    flags.DEFINE_integer(
        name="datasets_num_parallel_batches",
        default=None,
        help=help_wrap(
            "Determines how many batches to process in parallel when using "
            "map and batch from tf.data.")
    )

  return key_flags
