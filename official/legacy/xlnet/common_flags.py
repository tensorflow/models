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

"""Common flags used in XLNet model."""

from absl import flags

flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be "
    "either the name used when creating the Cloud TPU, or a "
    "url like grpc://ip.address.of.tpu:8470.")
flags.DEFINE_bool(
    "use_tpu", default=True, help="Use TPUs rather than plain CPUs.")
flags.DEFINE_string("tpu_topology", "2x2", help="TPU topology.")
flags.DEFINE_integer(
    "num_core_per_host", default=8, help="number of cores per host")

flags.DEFINE_string("model_dir", default=None, help="Estimator model_dir.")
flags.DEFINE_string(
    "init_checkpoint",
    default=None,
    help="Checkpoint path for initializing the model.")
flags.DEFINE_bool(
    "init_from_transformerxl",
    default=False,
    help="Init from a transformerxl model checkpoint. Otherwise, init from the "
    "entire model checkpoint.")

# Optimization config
flags.DEFINE_float("learning_rate", default=1e-4, help="Maximum learning rate.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping value.")
flags.DEFINE_float("weight_decay_rate", default=0.0, help="Weight decay rate.")

# lr decay
flags.DEFINE_integer(
    "warmup_steps", default=0, help="Number of steps for linear lr warmup.")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon.")
flags.DEFINE_float(
    "lr_layer_decay_rate",
    default=1.0,
    help="Top layer: lr[L] = FLAGS.learning_rate."
    "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float(
    "min_lr_ratio", default=0.0, help="Minimum ratio learning rate.")

# Training config
flags.DEFINE_integer(
    "train_batch_size",
    default=16,
    help="Size of the train batch across all hosts.")
flags.DEFINE_integer(
    "train_steps", default=100000, help="Total number of training steps.")
flags.DEFINE_integer(
    "iterations", default=1000, help="Number of iterations per repeat loop.")

# Data config
flags.DEFINE_integer(
    "seq_len", default=0, help="Sequence length for pretraining.")
flags.DEFINE_integer(
    "reuse_len",
    default=0,
    help="How many tokens to be reused in the next batch. "
    "Could be half of `seq_len`.")
flags.DEFINE_bool("uncased", False, help="Use uncased inputs or not.")
flags.DEFINE_bool(
    "bi_data",
    default=False,
    help="Use bidirectional data streams, "
    "i.e., forward & backward.")
flags.DEFINE_integer("n_token", 32000, help="Vocab size")

# Model config
flags.DEFINE_integer("mem_len", default=0, help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False, help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length")

flags.DEFINE_integer("n_layer", default=6, help="Number of layers.")
flags.DEFINE_integer("d_model", default=32, help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=32, help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=4, help="Number of attention heads.")
flags.DEFINE_integer(
    "d_head", default=8, help="Dimension of each attention head.")
flags.DEFINE_integer(
    "d_inner",
    default=32,
    help="Dimension of inner hidden size in positionwise "
    "feed-forward.")
flags.DEFINE_float("dropout", default=0.1, help="Dropout rate.")
flags.DEFINE_float("dropout_att", default=0.1, help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False, help="Untie r_w_bias and r_r_bias")
flags.DEFINE_string(
    "ff_activation",
    default="relu",
    help="Activation type used in position-wise feed-forward.")
flags.DEFINE_string(
    "strategy_type",
    default="tpu",
    help="Activation type used in position-wise feed-forward.")
flags.DEFINE_bool("use_bfloat16", False, help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum(
    "init_method",
    default="normal",
    enum_values=["normal", "uniform"],
    help="Initialization method.")
flags.DEFINE_float(
    "init_std", default=0.02, help="Initialization std when init is normal.")
flags.DEFINE_float(
    "init_range", default=0.1, help="Initialization std when init is uniform.")

flags.DEFINE_integer(
    "test_data_size", default=12048, help="Number of test data samples.")
flags.DEFINE_string(
    "train_tfrecord_path",
    default=None,
    help="Path to preprocessed training set tfrecord.")
flags.DEFINE_string(
    "test_tfrecord_path",
    default=None,
    help="Path to preprocessed test set tfrecord.")
flags.DEFINE_integer(
    "test_batch_size",
    default=16,
    help="Size of the test batch across all hosts.")
flags.DEFINE_integer(
    "save_steps", default=1000, help="Number of steps for saving checkpoint.")
FLAGS = flags.FLAGS
