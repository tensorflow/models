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
            trainable=False)

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
    if not isinstance(tensors, dict):
      names = [str(i) for i in range(len(tensors))]
      tensors = collections.OrderedDict(zip(names, tensors))
    if not isinstance(tensors, collections.OrderedDict):
      tensors = collections.OrderedDict(
          sorted(tensors.items(), key=lambda t: t[0]))
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
    with tf.name_scope('Gather'):
      index_bound_assert = tf.Assert(
          tf.less(
              tf.to_int64(tf.reduce_max(indices)),
              tf.minimum(self.get_num_adds(), self._buffer_size)),
          ['Index out of bounds.'])
      with tf.control_dependencies([index_bound_assert]):
        indices = tf.convert_to_tensor(indices)

      batch = []
      for key in keys:
        batch.append(tf.gather(self._tensors[key], indices, name=key))
      return batch

  def gather_nstep(self, num_steps, indices, keys=None):
    """Returns elements at the specified indices from the buffer.

    Args:
      num_steps: (integer) length of trajectories to return.
      indices: (list of rank num_steps int Tensor) indices in the buffer to
        retrieve elements from for multiple trajectories. Each Tensor in the
        list represents the indices for a trajectory.
      keys: List of keys of tensors to retrieve. If None retrieve all.
    Returns:
      A list of list-of-tensors, where each element in the list is obtained by
        indexing one of the tensors in the buffer.
    Raises:
      ValueError: If gather is called before calling the add function.
      tf.errors.InvalidArgumentError: If indices are bigger than the number of
        items in the buffer.
    """
    if not self._tensors:
      raise ValueError('The add function must be called before calling gather.')
    if keys is None:
      keys = self._tensors.keys()
    with tf.name_scope('Gather'):
      index_bound_assert = tf.Assert(
          tf.less_equal(
              tf.to_int64(tf.reduce_max(indices) + num_steps),
              self.get_num_adds()),
          ['Trajectory indices go out of bounds.'])
      with tf.control_dependencies([index_bound_assert]):
        indices = tf.map_fn(
            lambda x: tf.mod(tf.range(x, x + num_steps), self._buffer_size),
            indices,
            dtype=tf.int64)

      batch = []
      for key in keys:

        def SampleTrajectories(trajectory_indices, key=key,
                               num_steps=num_steps):
          trajectory_indices.set_shape([num_steps])
          return tf.gather(self._tensors[key], trajectory_indices, name=key)

        batch.append(tf.map_fn(SampleTrajectories, indices,
                               dtype=self._tensors[key].dtype))
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
