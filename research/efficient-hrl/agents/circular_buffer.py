# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""A circular buffer where each element is a list of tensors.

Each element of the buffer is a list of tensors. An example use case is a replay
buffer in reinforcement learning, where each element is a list of tensors
representing the state, action, reward etc.

New elements are added sequentially, and once the buffer is full, we
start overwriting them in a circular fashion. Reading does not remove any
elements, only adding new elements does.

self._tensors: a dict of variables to serve as buffers
self.num_add_cs: number of tensors added in total
self.add(): adds a dict of tensors (no batch size dim) to the buffers
"""

import collections
import numpy as np
import tensorflow as tf

import gin.tf


@gin.configurable
class CircularBuffer(object):
  """A circular buffer where each element is a list of tensors."""

  def __init__(self, buffer_size=1000, scope='replay_buffer'):
    """Circular buffer of list of tensors.

    Args:
      buffer_size: (integer) maximum number of tensor lists the buffer can hold.
      scope: (string) variable scope for creating the variables.
    """
    self._buffer_size = np.int64(buffer_size)
    self._scope = scope
    self._tensors = collections.OrderedDict()
    with tf.variable_scope(self._scope):
      self._num_adds = tf.Variable(0, dtype=tf.int64, name='num_adds')
    self._num_adds_cs = tf.contrib.framework.CriticalSection(name='num_adds')

  @property
  def buffer_size(self):
    return self._buffer_size

  @property
  def scope(self):
    return self._scope

  @property
  def num_adds(self):
    return self._num_adds

  def _create_variables(self, tensors):
    with tf.variable_scope(self._scope):
      for name in tensors.keys():
        tensor = tensors[name]
        self._tensors[name] = tf.get_variable(
            name='BufferVariable_' + name,
            shape=[self._buffer_size] + tensor.get_shape().as_list(),
            dtype=tensor.dtype,
            trainable=False,
            initializer=(
              tf.initializers.constant({
                tf.bool: False,
                tf.int32: -1,
                tf.int64: -1,
              }.get(tensor.dtype, np.nan))
            ))  # should not use the default uniform normalizer, which makes no sense in a buffer.

  def _validate(self, tensors):
    """Validate shapes of tensors."""
    if len(tensors) != len(self._tensors):
      raise ValueError('Expected tensors to have %d elements. Received %d '
                       'instead.' % (len(self._tensors), len(tensors)))
    if self._tensors.keys() != tensors.keys():
      raise ValueError('The keys of tensors should be the always the same.'
                       'Received %s instead %s.' %
                       (tensors.keys(), self._tensors.keys()))
    for name, tensor in tensors.items():
      if tensor.get_shape().as_list() != self._tensors[
          name].get_shape().as_list()[1:]:
        raise ValueError('Tensor %s has incorrect shape.' % name)
      if not tensor.dtype.is_compatible_with(self._tensors[name].dtype):
        raise ValueError(
            'Tensor %s has incorrect data type. Expected %s, received %s' %
            (name, self._tensors[name].read_value().dtype, tensor.dtype))

  def add(self, tensors):
    """Adds an element (list/tuple/dict of tensors) to the buffer.

    Args:
      tensors: (list/tuple/dict of tensors) to be added to the buffer.
    Returns:
      An add operation that adds the input `tensors` to the buffer. Similar to
        an enqueue_op.
    Raises:
      ValueError: If the shapes and data types of input `tensors' are not the
        same across calls to the add function.
    """
    return self.maybe_add(tensors, True)

  def maybe_add(self, tensors, condition):
    """Adds an element (tensors) to the buffer based on the condition..

    Args:
      tensors: (list/tuple of tensors) to be added to the buffer.
      condition: A boolean Tensor controlling whether the tensors would be added
        to the buffer or not.
    Returns:
      An add operation that adds the input `tensors` to the buffer. Similar to
        an maybe_enqueue_op.
    Raises:
      ValueError: If the shapes and data types of input `tensors' are not the
        same across calls to the add function.
    """
    # Convert tensors to an ordered dict. If given a list, keys are natural numbers.
    if isinstance(tensors, list):
      names = [str(i) for i in range(len(tensors))]
      tensors = collections.OrderedDict(zip(names, tensors))
    elif isinstance(tensors, dict) and not isinstance(tensors, collections.OrderedDict):
      tensors = collections.OrderedDict(
          sorted(tensors.items(), key=lambda t: t[0]))
    elif isinstance(tensors, collections.OrderedDict):
      pass
    else:
      raise NotImplementedError()

    if not self._tensors:
      self._create_variables(tensors)
    else:
      self._validate(tensors)

    #@tf.critical_section(self._position_mutex)
    def _increment_num_adds():
      # Adding 0 to the num_adds variable is a trick to read the value of the
      # variable and return a read-only tensor. Doing this in a critical
      # section allows us to capture a snapshot of the variable that will
      # not be affected by other threads updating num_adds.
      return self._num_adds.assign_add(1) + 0
    def _add():
      num_adds_inc = self._num_adds_cs.execute(_increment_num_adds)
      current_pos = tf.mod(num_adds_inc - 1, self._buffer_size)
      update_ops = []
      for name in self._tensors.keys():
        update_ops.append(
            tf.scatter_update(self._tensors[name], current_pos, tensors[name]))
      return tf.group(*update_ops)

    return tf.contrib.framework.smart_cond(condition, _add, tf.no_op)

  def get_random_batch(self, batch_size, keys=None, num_steps=1):
    """Samples a batch of tensors from the buffer with replacement.

    Args:
      batch_size: (integer) number of elements to sample.
      keys: List of keys of tensors to retrieve. If None retrieve all.
      num_steps: (integer) length of trajectories to return. If > 1 will return
        a list of lists, where each internal list represents a trajectory of
        length num_steps.
    Returns:
      A list of tensors, where each element in the list is a batch sampled from
        one of the tensors in the buffer.
    Raises:
      ValueError: If get_random_batch is called before calling the add function.
      tf.errors.InvalidArgumentError: If this operation is executed before any
        items are added to the buffer.
    """
    if not self._tensors:
      raise ValueError('The add function must be called before get_random_batch.')
    if keys is None:
      keys = self._tensors.keys()
    else:
      assert set(keys) <= set(self._tensors.keys())

    latest_start_index = self.get_num_adds() - num_steps + 1
    empty_buffer_assert = tf.Assert(
        tf.greater(latest_start_index, 0),
        ['Not enough elements have been added to the buffer.'])
    with tf.control_dependencies([empty_buffer_assert]):
      max_index = tf.minimum(self._buffer_size, latest_start_index)
      indices = tf.random_uniform(
          [batch_size],
          minval=0,
          maxval=max_index,
          dtype=tf.int64)
      if num_steps == 1:
        return self.gather(indices, keys)
      else:
        return self.gather_nstep(num_steps, indices, keys)

  def get_valid_start_indices(self, num_steps, episode_end_key='done',
                              episode_start_offset=0, num_subsample=None):
    """Get all indices s.t. time steps [x - episode_start_offset, x + num_steps)
    lies within the same episode (including the terminal state).
    """
    assert episode_end_key in self._tensors
    episode_end_indices = tf.squeeze(
      tf.where(self._tensors[episode_end_key], name='episode_end_indices'),
      name='episode_end_indices_flat',
      axis=1,
    )  # 1D

    # def compute_start_indices(_episode_end_indices):
    #   nonlocal num_steps
    #   if len(_episode_end_indices) < 2:
    #     indices = np.zeros(shape=(), dtype=_episode_end_indices.dtype)
    #   else:
    #     indices = np.concatenate([
    #       np.arange(_episode_end_indices[i] + 1 + episode_start_offset,  # +1: not including prev done
    #                 _episode_end_indices[i + 1] - num_steps + 2)  # +2: can include next done
    #       for i in range(len(_episode_end_indices) - 1)
    #     ])
    #   return indices
    #
    # start_indices = tf.py_func(
    #   compute_start_indices,
    #   inp=[episode_end_indices],
    #   Tout=tf.int64,
    # )
    # start_indices.set_shape([None])

    starts = tf.cond(
      tf.size(episode_end_indices) >= 2,
      lambda: episode_end_indices[:-1] + 1 + episode_start_offset,
      lambda: tf.zeros([0], dtype=episode_end_indices.dtype),
    )  # [N]
    ends = tf.cond(
      tf.size(episode_end_indices) >= 2,
      lambda: episode_end_indices[1:] - num_steps + 2,
      lambda: tf.zeros([0], dtype=episode_end_indices.dtype),
    )# [N]
    # Sample indices between in some range(start, end)
    lengths = ends - starts  # [N]
    nonempties = tf.where(lengths > 0)[:, 0]  # indices of non-empty episodes
    starts, ends, lengths = map(lambda x: tf.gather(x, nonempties),
                                [starts, ends, lengths])

    total_length = tf.reduce_sum(lengths)
    cumsum_lengths = tf.cumsum(lengths, exclusive=True)  # [N]: [0, l1, l1 + l2, ...]
    if num_subsample is None:
      subindices = tf.range(total_length)
    else:
      subindices = tf.random.uniform(
        shape=[num_subsample], minval=0, maxval=total_length,
        dtype=tf.int64, name='sampled_subindices')  # [num_subsample]

    within_episode_indices = tf.subtract(tf.expand_dims(subindices, 1),
                                         tf.expand_dims(cumsum_lengths, 0))  # [N, num_subsample]
    within_episode_indices = tf.where(
      within_episode_indices < 0,
      (1 + tf.reduce_max(within_episode_indices)) * tf.ones_like(within_episode_indices),
      within_episode_indices,
    )
    matching_episodes = tf.argmin(within_episode_indices, axis=1)  # [num_subsample]
    within_matching_episodes_indices = tf.reduce_min(within_episode_indices, axis=1)  # [num_subsample]
    start_indices = tf.add(
      tf.cast(tf.gather(starts, matching_episodes), tf.int64),
      tf.cast(within_matching_episodes_indices, tf.int64),
      name='start_indices',
    )  # [num_subsample]

    if 'env_time_step' in self._tensors:
      # TODO: to save time, maybe only check the last time step
      start_time_steps = tf.gather(self._tensors['env_time_step'], start_indices,
                                   name='start_time_steps')  # [N]
      expected_time_steps = tf.add(
        tf.cast(tf.expand_dims(start_time_steps, axis=1), tf.int64), # [N,1]
        tf.expand_dims(tf.range(num_steps, dtype=tf.int64), axis=0),  # [1, num_steps]
        name='expected_time_steps',
      )  # [N, num_steps]
      all_indices = (tf.expand_dims(start_indices, 1) +
                     tf.expand_dims(tf.range(num_steps, dtype=tf.int64), 0))  # [len(start_indices), num_steps]
      actual_time_steps = tf.gather_nd(
        tf.cast(self._tensors['env_time_step'], tf.int64), # [buffer_size]
        tf.expand_dims(all_indices, 2, name='all_indices'),  # [len(start_indices), num_steps, 1]
        name='actual_time_steps',
      )
      # Only grab a start_index if its following time steps are consecutive.
      diffs = tf.reduce_sum(tf.abs(expected_time_steps - actual_time_steps), axis=1)
      good = tf.where(tf.equal(diffs, 0))[:, 0]  # don't use diffs == 0
      good_ratio = tf.div(
        tf.cast(tf.size(good), tf.float32),
        tf.cast(tf.size(diffs), tf.float32),
        name='good_ratio',
      )
      with tf.control_dependencies([
        tf.cond(good_ratio < 0.9, lambda: tf.print('good_ratio:', good_ratio), lambda: tf.constant(False))
      ]):
        start_indices = tf.gather(start_indices, good)

      # assert_consecutive_steps = tf.assert_equal(
      #   expected_time_steps, actual_time_steps,
      #   data=[tf.gather(expected_time_steps, good), tf.gather(actual_time_steps, good)],
      #   summarize=100,
      #   name='assert_consecutive_steps')

      # with tf.control_dependencies([assert_consecutive_steps]):
      #   start_indices = start_indices + 0
    return start_indices


  def gather(self, indices, keys=None):
    """Returns elements at the specified indices from the buffer.

    Args:
      indices: (list of integers or rank 1 int Tensor) indices in the buffer to
        retrieve elements from.
      keys: List of keys of tensors to retrieve. If None retrieve all.
    Returns:
      A list of tensors, where each element in the list is obtained by indexing
        one of the tensors in the buffer.
    Raises:
      ValueError: If gather is called before calling the add function.
      tf.errors.InvalidArgumentError: If indices are bigger than the number of
        items in the buffer.
    """
    if not self._tensors:
      raise ValueError('The add function must be called before calling gather.')
    if keys is None:
      keys = self._tensors.keys()
    else:
      assert set(keys) <= set(self._tensors.keys())
    with tf.name_scope('Gather'):
      index_bound_assert = tf.Assert(
          tf.less(
              tf.to_int64(tf.reduce_max(indices)),
              tf.minimum(self.get_num_adds(), self._buffer_size)),
          ['Index out of bounds.'])
      with tf.control_dependencies([index_bound_assert]):
        indices = tf.convert_to_tensor(indices)

      batch = collections.OrderedDict({
        key: tf.gather(self._tensors[key], indices, name=key)
        for key in keys
      })
      return batch

  def gather_nstep(self, num_steps, start_indices, keys=None,
                   check_indices_validity=True):
    """Returns consecutive slices of buffer tensors.

    Args:
      # TODO: make num_steps a tensor/variable
      num_steps: (integer) length of trajectories to return.
      start_indices: a 1D integer tensor that mark the beginning of
        sequences of length num_steps. Indices are relative to the very first
        experience inserted to the buffer, and thus can be greater than
        self._buffer_size.
      keys: List of keys of tensors to retrieve. If None retrieve all, and order
        them by self._tensors.keys() (OrderedDict keys)
      check_indices_validity: If True, check if the indices are valid. If your
        start_indices are computed from self.get_valid_start_indices(), you
        may set it to False to avoid re-computation.
    Returns:
      A list of tensors of size [len(indices), num_steps, ...], each tensor
      corresponding to a key.
      (If num_steps == 1, the size becomes [len(indices), ...])
    Raises:
      ValueError: If gather is called before calling the add function.
      tf.errors.InvalidArgumentError: If indices are bigger than the number of
        items in the buffer.
    """
    if not self._tensors:
      raise ValueError('The add function must be called before calling gather.')
    if keys is None:
      keys = self._tensors.keys()
    else:
      assert set(keys) <= set(self._tensors.keys())

    with tf.name_scope('Gather_' + '_'.join(keys)):
      if check_indices_validity:
        def py_set_inclusion(tensor1, tensor2):
          return set(tensor1) <= set(tensor2)

        valid_indices = self.get_valid_start_indices(num_steps)
        index_bound_assert = tf.Assert(
          tf.py_func(
            py_set_inclusion,
            inp=[start_indices, valid_indices],
            Tout=tf.bool,
            name='start_indices_subset_of_valid_indices',
          ),
          ['Trajectory start_indices go out of bounds.',
           start_indices, valid_indices],
          summarize=10,
          name='assert_valid_start_indices',
        )
      else:
        index_bound_assert = tf.constant(0.)  # do nothing
      with tf.control_dependencies([index_bound_assert]):
        all_indices = (tf.expand_dims(start_indices, 1) +
                       tf.expand_dims(tf.range(num_steps, dtype=tf.int64), 0))  # [len(start_indices), num_steps]
        if num_steps != 1:
          all_indices = tf.expand_dims(all_indices, 2, name='all_indices')  # [len(start_indices), num_steps, 1]

      batch = collections.OrderedDict({
        key: tf.gather_nd(self._tensors[key], all_indices, name=key)
        for key in keys
      })
      return batch

  def get_position(self):
    """Returns the position at which the last element was added.

    Returns:
      An int tensor representing the index at which the last element was added
        to the buffer or -1 if no elements were added.
    """
    return tf.cond(self.get_num_adds() < 1,
                   lambda: self.get_num_adds() - 1,
                   lambda: tf.mod(self.get_num_adds() - 1, self._buffer_size))

  def get_num_adds(self):
    """Returns the number of additions to the buffer.

    Returns:
      An int tensor representing the number of elements that were added.
    """
    def num_adds():
      return self._num_adds.value()

    return self._num_adds_cs.execute(num_adds)

  def get_num_tensors(self):
    """Returns the number of tensors (slots) in the buffer."""
    return len(self._tensors)


def test_gather_nsteps():
  sess = tf.Session()
  buffer = CircularBuffer(buffer_size=10)
  experience_T = {
    'state': tf.placeholder(dtype=tf.float32, shape=()),
    'action': tf.placeholder(dtype=tf.float32, shape=()),
    'reward': tf.placeholder(dtype=tf.float32, shape=()),
    'done': tf.placeholder(dtype=tf.bool, shape=()),
    'env_time_step': tf.placeholder(dtype=tf.int32, shape=()),
  }
  buffer.maybe_add(experience_T, False)  # create buffer vars at python-level
  sess.run(tf.global_variables_initializer())  # init buffer vars
  add_op = buffer.maybe_add(experience_T, True)


  # Test addition
  def add_one_episode():
    T = 3
    for i in range(T):
      feed_dict = {
        experience_T['state']: np.array(i),
        experience_T['action']: np.array(i + 0.5),
        experience_T['reward']: np.array(i + 0.1),
        experience_T['done']: np.array(i==(T-1)),
        experience_T['env_time_step']: np.array(i),
      }
      sess.run(add_op, feed_dict=feed_dict)

  def print_dict(D):
    print('\n'.join([str((k, v)) for k, v in D.items()]))

  for _ in range(4):
    add_one_episode()
    print_dict(sess.run(buffer._tensors))
    print('\n')
  # Last dones should be [False,  True,  True, False, False,
  # True, False, False, True, False]

  # Test start_indices computation
  for length in [2, 3]:
    print('Length %d start indices' % length,
          sess.run(buffer.get_valid_start_indices(length)))
    # [3 4 6 7]
    # [3 6]

  # Test gather_nsteps
  for length, indices in zip([1, 2, 3], [[3], [3, 4], [3, 6]]):
    print('Length %d batches' % length)
    batch = sess.run(buffer.gather_nstep(
      num_steps=length, start_indices=tf.constant(indices, dtype=tf.int64)))
    print_dict(batch)


def test_maze():
  from train_v2 import tf_rand_sample
  sess = tf.Session()
  buffer = CircularBuffer(buffer_size=200000)
  experience_T = {
    'step': tf.placeholder(dtype=tf.float32, shape=()),
    'done': tf.placeholder(dtype=tf.bool, shape=()),
  }
  buffer.maybe_add(experience_T, False)  # create buffer vars at python-level
  sess.run(tf.global_variables_initializer())  # init buffer vars
  add_op = buffer.maybe_add(experience_T, True)
  get_op = buffer.gather_nstep(
    num_steps=10,
    start_indices=tf_rand_sample(
      buffer.get_valid_start_indices(num_steps=10, episode_start_offset=1),
      size=1,
    ),
    keys=['step'],
  )['step']

  def add_one_episode(T=501):
    for i in range(T):
      sess.run(add_op, {experience_T['step']: i, experience_T['done']: (i == T-1)})

  def get_one_segment():
    return sess.run(get_op)

  def print_dict(D):
    print('\n'.join([str((k, v)) for k, v in D.items()]))


  for _ in range(5):
    add_one_episode()

  for _ in range(3000):
    add_one_episode()
    print(get_one_segment())
    # print_dict(sess.run(buffer._tensors))
    print('\n')



if __name__ == '__main__':
  test_gather_nsteps()
