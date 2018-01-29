# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# ==============================================================================
"""Memory module for storing "nearest neighbors".

Implements a key-value memory for generalized one-shot learning
as described in the paper
"Learning to Remember Rare Events"
by Lukasz Kaiser, Ofir Nachum, Aurko Roy, Samy Bengio,
published as a conference paper at ICLR 2017.
"""

import numpy as np
from six.moves import xrange
import tensorflow as tf


class Memory(object):
  """Memory module."""

  def __init__(self, key_dim, memory_size, vocab_size,
               choose_k=256, alpha=0.1, correct_in_top=1, age_noise=8.0,
               var_cache_device='', nn_device=''):
    self.key_dim = key_dim
    self.memory_size = memory_size
    self.vocab_size = vocab_size
    self.choose_k = min(choose_k, memory_size)
    self.alpha = alpha
    self.correct_in_top = correct_in_top
    self.age_noise = age_noise
    self.var_cache_device = var_cache_device  # Variables are cached here.
    self.nn_device = nn_device  # Device to perform nearest neighbour matmul.

    caching_device = var_cache_device if var_cache_device else None
    self.update_memory = tf.constant(True)  # Can be fed "false" if needed.
    self.mem_keys = tf.get_variable(
        'memkeys', [self.memory_size, self.key_dim], trainable=False,
        initializer=tf.random_uniform_initializer(-0.0, 0.0),
        caching_device=caching_device)
    self.mem_vals = tf.get_variable(
        'memvals', [self.memory_size], dtype=tf.int32, trainable=False,
        initializer=tf.constant_initializer(0, tf.int32),
        caching_device=caching_device)
    self.mem_age = tf.get_variable(
        'memage', [self.memory_size], dtype=tf.float32, trainable=False,
        initializer=tf.constant_initializer(0.0), caching_device=caching_device)
    self.recent_idx = tf.get_variable(
        'recent_idx', [self.vocab_size], dtype=tf.int32, trainable=False,
        initializer=tf.constant_initializer(0, tf.int32))

    # variable for projecting query vector into memory key
    self.query_proj = tf.get_variable(
        'memory_query_proj', [self.key_dim, self.key_dim], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(0, 0.01),
        caching_device=caching_device)

  def get(self):
    return self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx

  def set(self, k, v, a, r=None):
    return tf.group(
        self.mem_keys.assign(k),
        self.mem_vals.assign(v),
        self.mem_age.assign(a),
        (self.recent_idx.assign(r) if r is not None else tf.group()))

  def clear(self):
    return tf.variables_initializer([self.mem_keys, self.mem_vals, self.mem_age,
                                     self.recent_idx])

  def get_hint_pool_idxs(self, normalized_query):
    """Get small set of idxs to compute nearest neighbor queries on.

    This is an expensive look-up on the whole memory that is used to
    avoid more expensive operations later on.

    Args:
      normalized_query: A Tensor of shape [None, key_dim].

    Returns:
      A Tensor of shape [None, choose_k] of indices in memory
      that are closest to the queries.

    """
    # look up in large memory, no gradients
    with tf.device(self.nn_device):
      similarities = tf.matmul(tf.stop_gradient(normalized_query),
                               self.mem_keys, transpose_b=True, name='nn_mmul')
    _, hint_pool_idxs = tf.nn.top_k(
        tf.stop_gradient(similarities), k=self.choose_k, name='nn_topk')
    return hint_pool_idxs

  def make_update_op(self, upd_idxs, upd_keys, upd_vals,
                     batch_size, use_recent_idx, intended_output):
    """Function that creates all the update ops."""
    mem_age_incr = self.mem_age.assign_add(tf.ones([self.memory_size],
                                                   dtype=tf.float32))
    with tf.control_dependencies([mem_age_incr]):
      mem_age_upd = tf.scatter_update(
          self.mem_age, upd_idxs, tf.zeros([batch_size], dtype=tf.float32))

    mem_key_upd = tf.scatter_update(
        self.mem_keys, upd_idxs, upd_keys)
    mem_val_upd = tf.scatter_update(
        self.mem_vals, upd_idxs, upd_vals)

    if use_recent_idx:
      recent_idx_upd = tf.scatter_update(
          self.recent_idx, intended_output, upd_idxs)
    else:
      recent_idx_upd = tf.group()

    return tf.group(mem_age_upd, mem_key_upd, mem_val_upd, recent_idx_upd)

  def query(self, query_vec, intended_output, use_recent_idx=True):
    """Queries memory for nearest neighbor.

    Args:
      query_vec: A batch of vectors to query (embedding of input to model).
      intended_output: The values that would be the correct output of the
        memory.
      use_recent_idx: Whether to always insert at least one instance of a
        correct memory fetch.

    Returns:
      A tuple (result, mask, teacher_loss).
      result: The result of the memory look up.
      mask: The affinity of the query to the result.
      teacher_loss: The loss for training the memory module.
    """

    batch_size = tf.shape(query_vec)[0]
    output_given = intended_output is not None

    # prepare query for memory lookup
    query_vec = tf.matmul(query_vec, self.query_proj)
    normalized_query = tf.nn.l2_normalize(query_vec, dim=1)

    hint_pool_idxs = self.get_hint_pool_idxs(normalized_query)

    if output_given and use_recent_idx:  # add at least one correct memory
      most_recent_hint_idx = tf.gather(self.recent_idx, intended_output)
      hint_pool_idxs = tf.concat(
          axis=1,
          values=[hint_pool_idxs, tf.expand_dims(most_recent_hint_idx, 1)])
    choose_k = tf.shape(hint_pool_idxs)[1]

    with tf.device(self.var_cache_device):
      # create small memory and look up with gradients
      my_mem_keys = tf.stop_gradient(tf.gather(self.mem_keys, hint_pool_idxs,
                                               name='my_mem_keys_gather'))
      similarities = tf.matmul(tf.expand_dims(normalized_query, 1),
                               my_mem_keys, adjoint_b=True, name='batch_mmul')
      hint_pool_sims = tf.squeeze(similarities, [1], name='hint_pool_sims')
      hint_pool_mem_vals = tf.gather(self.mem_vals, hint_pool_idxs,
                                     name='hint_pool_mem_vals')
    # Calculate softmax mask on the top-k if requested.
    # Softmax temperature. Say we have K elements at dist x and one at (x+a).
    # Softmax of the last is e^tm(x+a)/Ke^tm*x + e^tm(x+a) = e^tm*a/K+e^tm*a.
    # To make that 20% we'd need to have e^tm*a ~= 0.2K, so tm = log(0.2K)/a.
    softmax_temp = max(1.0, np.log(0.2 * self.choose_k) / self.alpha)
    mask = tf.nn.softmax(hint_pool_sims[:, :choose_k - 1] * softmax_temp)

    # prepare hints from the teacher on hint pool
    teacher_hints = tf.to_float(
        tf.abs(tf.expand_dims(intended_output, 1) - hint_pool_mem_vals))
    teacher_hints = 1.0 - tf.minimum(1.0, teacher_hints)

    teacher_vals, teacher_hint_idxs = tf.nn.top_k(
        hint_pool_sims * teacher_hints, k=1)
    neg_teacher_vals, _ = tf.nn.top_k(
        hint_pool_sims * (1 - teacher_hints), k=1)

    # bring back idxs to full memory
    teacher_idxs = tf.gather(
        tf.reshape(hint_pool_idxs, [-1]),
        teacher_hint_idxs[:, 0] + choose_k * tf.range(batch_size))

    # zero-out teacher_vals if there are no hints
    teacher_vals *= (
        1 - tf.to_float(tf.equal(0.0, tf.reduce_sum(teacher_hints, 1))))

    # prepare returned values
    nearest_neighbor = tf.to_int32(
        tf.argmax(hint_pool_sims[:, :choose_k - 1], 1))
    no_teacher_idxs = tf.gather(
        tf.reshape(hint_pool_idxs, [-1]),
        nearest_neighbor + choose_k * tf.range(batch_size))

    # we'll determine whether to do an update to memory based on whether
    # memory was queried correctly
    sliced_hints = tf.slice(teacher_hints, [0, 0], [-1, self.correct_in_top])
    incorrect_memory_lookup = tf.equal(0.0, tf.reduce_sum(sliced_hints, 1))

    # loss based on triplet loss
    teacher_loss = (tf.nn.relu(neg_teacher_vals - teacher_vals + self.alpha)
                    - self.alpha)

    with tf.device(self.var_cache_device):
      result = tf.gather(self.mem_vals, tf.reshape(no_teacher_idxs, [-1]))

    # prepare memory updates
    update_keys = normalized_query
    update_vals = intended_output

    fetched_idxs = teacher_idxs  # correctly fetched from memory
    with tf.device(self.var_cache_device):
      fetched_keys = tf.gather(self.mem_keys, fetched_idxs, name='fetched_keys')
      fetched_vals = tf.gather(self.mem_vals, fetched_idxs, name='fetched_vals')

    # do memory updates here
    fetched_keys_upd = update_keys + fetched_keys  # Momentum-like update
    fetched_keys_upd = tf.nn.l2_normalize(fetched_keys_upd, dim=1)
    # Randomize age a bit, e.g., to select different ones in parallel workers.
    mem_age_with_noise = self.mem_age + tf.random_uniform(
        [self.memory_size], - self.age_noise, self.age_noise)

    _, oldest_idxs = tf.nn.top_k(mem_age_with_noise, k=batch_size, sorted=False)

    with tf.control_dependencies([result]):
      upd_idxs = tf.where(incorrect_memory_lookup,
                          oldest_idxs,
                          fetched_idxs)
      # upd_idxs = tf.Print(upd_idxs, [upd_idxs], "UPD IDX", summarize=8)
      upd_keys = tf.where(incorrect_memory_lookup,
                          update_keys,
                          fetched_keys_upd)
      upd_vals = tf.where(incorrect_memory_lookup,
                          update_vals,
                          fetched_vals)

    def make_update_op():
      return self.make_update_op(upd_idxs, upd_keys, upd_vals,
                                 batch_size, use_recent_idx, intended_output)

    update_op = tf.cond(self.update_memory, make_update_op, tf.no_op)

    with tf.control_dependencies([update_op]):
      result = tf.identity(result)
      mask = tf.identity(mask)
      teacher_loss = tf.identity(teacher_loss)

    return result, mask, tf.reduce_mean(teacher_loss)


class LSHMemory(Memory):
  """Memory employing locality sensitive hashing.

  Note: Not fully tested.
  """

  def __init__(self, key_dim, memory_size, vocab_size,
               choose_k=256, alpha=0.1, correct_in_top=1, age_noise=8.0,
               var_cache_device='', nn_device='',
               num_hashes=None, num_libraries=None):
    super(LSHMemory, self).__init__(
        key_dim, memory_size, vocab_size,
        choose_k=choose_k, alpha=alpha, correct_in_top=1, age_noise=age_noise,
        var_cache_device=var_cache_device, nn_device=nn_device)

    self.num_libraries = num_libraries or int(self.choose_k ** 0.5)
    self.num_per_hash_slot = max(1, self.choose_k // self.num_libraries)
    self.num_hashes = (num_hashes or
                       int(np.log2(self.memory_size / self.num_per_hash_slot)))
    self.num_hashes = min(max(self.num_hashes, 1), 20)
    self.num_hash_slots = 2 ** self.num_hashes

    # hashing vectors
    self.hash_vecs = [
        tf.get_variable(
            'hash_vecs%d' % i, [self.num_hashes, self.key_dim],
            dtype=tf.float32, trainable=False,
            initializer=tf.truncated_normal_initializer(0, 1))
        for i in xrange(self.num_libraries)]

    # map representing which hash slots map to which mem keys
    self.hash_slots = [
        tf.get_variable(
            'hash_slots%d' % i, [self.num_hash_slots, self.num_per_hash_slot],
            dtype=tf.int32, trainable=False,
            initializer=tf.random_uniform_initializer(maxval=self.memory_size,
                                                      dtype=tf.int32))
        for i in xrange(self.num_libraries)]

  def get(self):  # not implemented
    return self.mem_keys, self.mem_vals, self.mem_age, self.recent_idx

  def set(self, k, v, a, r=None):  # not implemented
    return tf.group(
        self.mem_keys.assign(k),
        self.mem_vals.assign(v),
        self.mem_age.assign(a),
        (self.recent_idx.assign(r) if r is not None else tf.group()))

  def clear(self):
    return tf.variables_initializer([self.mem_keys, self.mem_vals, self.mem_age,
                                     self.recent_idx] + self.hash_slots)

  def get_hash_slots(self, query):
    """Gets hashed-to buckets for batch of queries.

    Args:
      query: 2-d Tensor of query vectors.

    Returns:
      A list of hashed-to buckets for each hash function.
    """

    binary_hash = [
        tf.less(tf.matmul(query, self.hash_vecs[i], transpose_b=True), 0)
        for i in xrange(self.num_libraries)]
    hash_slot_idxs = [
        tf.reduce_sum(
            tf.to_int32(binary_hash[i]) *
            tf.constant([[2 ** i for i in xrange(self.num_hashes)]],
                        dtype=tf.int32), 1)
        for i in xrange(self.num_libraries)]
    return hash_slot_idxs

  def get_hint_pool_idxs(self, normalized_query):
    """Get small set of idxs to compute nearest neighbor queries on.

    This is an expensive look-up on the whole memory that is used to
    avoid more expensive operations later on.

    Args:
      normalized_query: A Tensor of shape [None, key_dim].

    Returns:
      A Tensor of shape [None, choose_k] of indices in memory
      that are closest to the queries.

    """
    # get hash of query vecs
    hash_slot_idxs = self.get_hash_slots(normalized_query)

    # grab mem idxs in the hash slots
    hint_pool_idxs = [
        tf.maximum(tf.minimum(
            tf.gather(self.hash_slots[i], idxs),
            self.memory_size - 1), 0)
        for i, idxs in enumerate(hash_slot_idxs)]

    return tf.concat(axis=1, values=hint_pool_idxs)

  def make_update_op(self, upd_idxs, upd_keys, upd_vals,
                     batch_size, use_recent_idx, intended_output):
    """Function that creates all the update ops."""
    base_update_op = super(LSHMemory, self).make_update_op(
        upd_idxs, upd_keys, upd_vals,
        batch_size, use_recent_idx, intended_output)

    # compute hash slots to be updated
    hash_slot_idxs = self.get_hash_slots(upd_keys)

    # make updates
    update_ops = []
    with tf.control_dependencies([base_update_op]):
      for i, slot_idxs in enumerate(hash_slot_idxs):
        # for each slot, choose which entry to replace
        entry_idx = tf.random_uniform([batch_size],
                                      maxval=self.num_per_hash_slot,
                                      dtype=tf.int32)
        entry_mul = 1 - tf.one_hot(entry_idx, self.num_per_hash_slot,
                                   dtype=tf.int32)
        entry_add = (tf.expand_dims(upd_idxs, 1) *
                     tf.one_hot(entry_idx, self.num_per_hash_slot,
                                dtype=tf.int32))

        mul_op = tf.scatter_mul(self.hash_slots[i], slot_idxs, entry_mul)
        with tf.control_dependencies([mul_op]):
          add_op = tf.scatter_add(self.hash_slots[i], slot_idxs, entry_add)
          update_ops.append(add_op)

    return tf.group(*update_ops)
