# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions for inference."""
import tensorflow as tf


def split_and_pad(strategy, batch_size, x):
  """Split and pad for interence."""
  per_replica_size = batch_size // strategy.num_replicas_in_sync

  def slice_fn(x, i):
    begin = min(x.shape[0], i * per_replica_size)
    end = min(x.shape[0], (i + 1) * per_replica_size)
    indices = tf.range(begin, end, dtype=tf.int32)
    return tf.gather(x, tf.pad(indices, [[0, per_replica_size - end + begin]]))

  # pylint: disable=g-long-lambda
  return tf.nest.map_structure(
      lambda x: strategy.experimental_distribute_values_from_function(
          lambda ctx: slice_fn(x, ctx.replica_id_in_sync_group)), x)
  # pylint: enable=g-long-lambda


def decode_logits(top_k, max_size, logits, default):
  """Get the span from logits."""
  logits = tf.transpose(logits, [0, 2, 1])
  values, indices = tf.math.top_k(logits, top_k)
  width = (
      tf.expand_dims(indices[:, 1, :], -2) -
      tf.expand_dims(indices[:, 0, :], -1))
  mask = tf.logical_and(width >= 0, width <= max_size)
  scores = (
      tf.expand_dims(values[:, 0, :], -1) + tf.expand_dims(values[:, 1, :], -2))
  scores = tf.where(mask, scores, -1e8)
  flat_indices = tf.argmax(tf.reshape(scores, (-1, top_k * top_k)), -1)
  begin = tf.gather(
      indices[:, 0, :], tf.math.floordiv(flat_indices, top_k), batch_dims=1)
  end = tf.gather(
      indices[:, 1, :], tf.math.mod(flat_indices, top_k), batch_dims=1)
  reduced_mask = tf.math.reduce_any(mask, [-1, -2])
  return (tf.where(reduced_mask, begin,
                   default), tf.where(reduced_mask, end, default),
          tf.math.reduce_max(scores, [-1, -2]))


@tf.function
def decode_answer(context, begin, end, token_offsets, end_limit):
  i = tf.gather(token_offsets, begin, batch_dims=1)
  j = tf.gather(token_offsets, tf.minimum(end + 1, end_limit), batch_dims=1)
  j = tf.where(end == end_limit, tf.cast(tf.strings.length(context), tf.int64),
               j)
  return tf.strings.substr(context, i, j - i)


def distributed_logits_fn(model, x):
  return model.distribute_strategy.run(
      lambda x: model(x, training=False), args=(x,))
