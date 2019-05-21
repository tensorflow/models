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
"""Functions specific to running TensorFlow on TPUs."""

import tensorflow as tf


# "local" is a magic word in the TPU cluster resolver; it informs the resolver
# to use the local CPU as the compute device. This is useful for testing and
# debugging; the code flow is ostensibly identical, but without the need to
# actually have a TPU on the other end.
LOCAL = "local"


def construct_scalar_host_call(metric_dict, model_dir, prefix=""):
  """Construct a host call to log scalars when training on TPU.

  Args:
    metric_dict: A dict of the tensors to be logged.
    model_dir: The location to write the summary.
    prefix: The prefix (if any) to prepend to the metric names.

  Returns:
    A tuple of (function, args_to_be_passed_to_said_function)
  """
  # type: (dict, str) -> (function, list)
  metric_names = list(metric_dict.keys())

  def host_call_fn(global_step, *args):
    """Training host call. Creates scalar summaries for training metrics.

    This function is executed on the CPU and should not directly reference
    any Tensors in the rest of the `model_fn`. To pass Tensors from the
    model to the `metric_fn`, provide as part of the `host_call`. See
    https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
    for more information.

    Arguments should match the list of `Tensor` objects passed as the second
    element in the tuple passed to `host_call`.

    Args:
      global_step: `Tensor with shape `[batch]` for the global_step
      *args: Remaining tensors to log.

    Returns:
      List of summary ops to run on the CPU host.
    """
    step = global_step[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          tf.contrib.summary.scalar(prefix + name, args[i][0], step=step)

        return tf.contrib.summary.all_summary_ops()

  # To log the current learning rate, and gradient norm for Tensorboard, the
  # summary op needs to be run on the host CPU via host_call. host_call
  # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
  # dimension. These Tensors are implicitly concatenated to
  # [params['batch_size']].
  global_step_tensor = tf.reshape(
      tf.compat.v1.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def embedding_matmul(embedding_table, values, mask, name="embedding_matmul"):
  """Performs embedding lookup via a matmul.

  The matrix to be multiplied by the embedding table Tensor is constructed
  via an implementation of scatter based on broadcasting embedding indices
  and performing an equality comparison against a broadcasted
  range(num_embedding_table_rows). All masked positions will produce an
  embedding vector of zeros.

  Args:
    embedding_table: Tensor of embedding table.
      Rank 2 (table_size x embedding dim)
    values: Tensor of embedding indices. Rank 2 (batch x n_indices)
    mask: Tensor of mask / weights. Rank 2 (batch x n_indices)
    name: Optional name scope for created ops

  Returns:
    Rank 3 tensor of embedding vectors.
  """

  with tf.name_scope(name):
    n_embeddings = embedding_table.get_shape().as_list()[0]
    batch_size, padded_size = values.shape.as_list()

    emb_idcs = tf.tile(
        tf.reshape(values, (batch_size, padded_size, 1)), (1, 1, n_embeddings))
    emb_weights = tf.tile(
        tf.reshape(mask, (batch_size, padded_size, 1)), (1, 1, n_embeddings))
    col_idcs = tf.tile(
        tf.reshape(tf.range(n_embeddings), (1, 1, n_embeddings)),
        (batch_size, padded_size, 1))
    one_hot = tf.where(
        tf.equal(emb_idcs, col_idcs), emb_weights,
        tf.zeros((batch_size, padded_size, n_embeddings)))

    return tf.tensordot(one_hot, embedding_table, 1)
