from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Configuration class."""

import bisect
from collections import deque
import cPickle
import heapq
import random

from absl import logging
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf


def tuple_to_record(tuple_, record_type):
  return record_type(**dict(zip(record_type.__slots__, tuple_)))


def make_record(type_name, attributes, defaults=None):
  """Factory for mutable record classes.

  A record acts just like a collections.namedtuple except slots are writable.
  One exception is that record classes are not equivalent to tuples or other
  record classes of the same length.

  Note, each call to `make_record` produces a unique type. Two calls will make
  different types even if `type_name` is the same each time.

  Args:
    type_name: Name of the record type to create.
    attributes: List of names of each record attribute. The order of the list
        is preserved.
    defaults: (optional) default values for attributes. A dict mapping attribute
        names to values.

  Returns:
    A new record type.

  Raises:
    ValueError: If,
        `defaults` is not a dict,
        `attributes` contains duplicate names,
        `defaults` keys are not contained in `attributes`.
  """
  if defaults is None:
    defaults = {}
  if not isinstance(defaults, dict):
    raise ValueError('defaults must be a dict.')
  attr_set = set(attributes)
  if len(attr_set) < len(attributes):
    raise ValueError('No duplicate attributes allowed.')
  if not set(defaults.keys()).issubset(attr_set):
    raise ValueError('Default attributes must be given in the attributes list.')

  class RecordClass(object):
    """A record type.

    Acts like mutable tuple with named slots.
    """
    __slots__ = list(attributes)
    _defaults = dict(defaults)

    def __init__(self, *args, **kwargs):
      if len(args) > len(self.__slots__):
        raise ValueError('Too many arguments. %s has length %d.'
                         % (type(self).__name__, len(self.__slots__)))
      for attr, val in self._defaults.items():
        setattr(self, attr, val)
      for i, arg in enumerate(args):
        setattr(self, self.__slots__[i], arg)
      for attr, val in kwargs.items():
        setattr(self, attr, val)
      for attr in self.__slots__:
        if not hasattr(self, attr):
          raise ValueError('Required attr "%s" is not set.' % attr)

    def __len__(self):
      return len(self.__slots__)

    def __iter__(self):
      for attr in self.__slots__:
        yield getattr(self, attr)

    def __getitem__(self, index):
      return getattr(self, self.__slots__[index])

    def __setitem__(self, index, value):
      return setattr(self, self.__slots__[index], value)

    def __eq__(self, other):
      # Types must be equal as well as values.
      return (isinstance(other, type(self))
              and all(a == b for a, b in zip(self, other)))

    def __str__(self):
      return '%s(%s)' % (
          type(self).__name__,
          ', '.join(attr + '=' + str(getattr(self, attr))
                    for attr in self.__slots__))

    def __repr__(self):
      return str(self)

  RecordClass.__name__ = type_name
  return RecordClass


# Making minibatches.
def stack_pad(tensors, pad_axes=None, pad_to_lengths=None, dtype=np.float32,
              pad_value=0):
  """Stack tensors along 0-th dim and pad them to be the same shape.

  Args:
    tensors: Any list of iterables (python list, numpy array, etc). Can be 1D
        or multi-D iterables.
    pad_axes: An int or list of ints. Axes to pad along.
    pad_to_lengths: Length in each dimension. If pad_axes was an int, this is an
        int or None. If pad_axes was a list of ints, this is a list of mixed int
        and None types with the same length, or None. A None length means the
        maximum length among the given tensors is used.
    dtype: Type of output numpy array. Defaults to np.float32.
    pad_value: Value to use for padding. Defaults to 0.

  Returns:
    Numpy array containing the tensors stacked along the 0-th dimension and
        padded along the specified dimensions.

  Raises:
    ValueError: If the tensors do not have equal shapes along non-padded
        dimensions.
  """
  tensors = [np.asarray(t) for t in tensors]
  max_lengths = [max(l) for l in zip(*[t.shape for t in tensors])]
  same_axes = dict(enumerate(max_lengths))
  if pad_axes is None:
    pad_axes = []
  if isinstance(pad_axes, six.integer_types):
    if pad_to_lengths is not None:
      max_lengths[pad_axes] = pad_to_lengths
    del same_axes[pad_axes]
  else:
    if pad_to_lengths is None:
      pad_to_lengths = [None] * len(pad_axes)
    for i, l in zip(pad_axes, pad_to_lengths):
      if l is not None:
        max_lengths[i] = l
      del same_axes[i]
  same_axes_items = same_axes.items()
  dest = np.full([len(tensors)] + max_lengths, pad_value, dtype=dtype)
  for i, t in enumerate(tensors):
    for j, l in same_axes_items:
      if t.shape[j] != l:
        raise ValueError(
            'Tensor at index %d does not have size %d along axis %d'
            % (i, l, j))
    dest[[i] + [slice(0, d) for d in t.shape]] = t
  return dest


class RandomQueue(deque):

  def __init__(self, capacity):
    super(RandomQueue, self).__init__([], capacity)
    self.capacity = capacity

  def random_sample(self, sample_size):
    idx = np.random.choice(len(self), sample_size)
    return [self[i] for i in idx]

  def push(self, item):
    # Append to right. Oldest element will be popped from left.
    self.append(item)


class MPQItemContainer(object):
  """Class for holding an item with its score.

  Defines a comparison function for use in the heap-queue.
  """

  def __init__(self, score, item, extra_data):
    self.item = item
    self.score = score
    self.extra_data = extra_data

  def __cmp__(self, other):
    assert isinstance(other, type(self))
    return cmp(self.score, other.score)

  def __iter__(self):
    """Allows unpacking like a tuple."""
    yield self.score
    yield self.item
    yield self.extra_data

  def __repr__(self):
    """String representation of this item.

    `extra_data` is not included in the representation. We are assuming that
    `extra_data` is not easily interpreted by a human (if it was, it should be
    hashable, like a string or tuple).

    Returns:
      String representation of `self`.
    """
    return str((self.score, self.item))

  def __str__(self):
    return repr(self)


class MaxUniquePriorityQueue(object):
  """A maximum priority queue where duplicates are not added.

  The top items by score remain in the queue. When the capacity is reached,
  the lowest scored item in the queue will be dropped.

  This implementation differs from a typical priority queue, in that the minimum
  score is popped, instead of the maximum. Largest scores remain stuck in the
  queue. This is useful for accumulating the best known items from a population.

  The items used to determine uniqueness must be hashable, but additional
  non-hashable data may be stored with each item.
  """

  def __init__(self, capacity):
    self.capacity = capacity
    self.heap = []
    self.unique_items = set()

  def push(self, score, item, extra_data=None):
    """Push an item onto the queue.

    If the queue is at capacity, the item with the smallest score will be
    dropped. Note that it is assumed each item has exactly one score. The same
    item with a different score will still be dropped.

    Args:
      score: Number used to prioritize items in the queue. Largest scores are
          kept in the queue.
      item: A hashable item to be stored. Duplicates of this item will not be
          added to the queue.
      extra_data: An extra (possible not hashable) data to store with the item.
    """
    if item in self.unique_items:
      return
    if len(self.heap) >= self.capacity:
      _, popped_item, _ = heapq.heappushpop(
          self.heap, MPQItemContainer(score, item, extra_data))
      self.unique_items.add(item)
      self.unique_items.remove(popped_item)
    else:
      heapq.heappush(self.heap, MPQItemContainer(score, item, extra_data))
      self.unique_items.add(item)

  def pop(self):
    """Pop the item with the lowest score.

    Returns:
      score: Item's score.
      item: The item that was popped.
      extra_data: Any extra data stored with the item.
    """
    if not self.heap:
      return ()
    score, item, extra_data = heapq.heappop(self.heap)
    self.unique_items.remove(item)
    return score, item, extra_data

  def get_max(self):
    """Peek at the item with the highest score.

    Returns:
      Same as `pop`.
    """
    if not self.heap:
      return ()
    score, item, extra_data = heapq.nlargest(1, self.heap)[0]
    return score, item, extra_data

  def get_min(self):
    """Peek at the item with the lowest score.

    Returns:
      Same as `pop`.
    """
    if not self.heap:
      return ()
    score, item, extra_data = heapq.nsmallest(1, self.heap)[0]
    return score, item, extra_data

  def random_sample(self, sample_size):
    """Randomly select items from the queue.

    This does not modify the queue.

    Items are drawn from a uniform distribution, and not weighted by score.

    Args:
      sample_size: Number of random samples to draw. The same item can be
          sampled multiple times.

    Returns:
      List of sampled items (of length `sample_size`). Each element in the list
      is a tuple: (item, extra_data).
    """
    idx = np.random.choice(len(self.heap), sample_size)
    return [(self.heap[i].item, self.heap[i].extra_data) for i in idx]

  def iter_in_order(self):
    """Iterate over items in the queue from largest score to smallest.

    Yields:
      item: Hashable item.
      extra_data: Extra data stored with the item.
    """
    for _, item, extra_data in heapq.nlargest(len(self.heap), self.heap):
      yield item, extra_data

  def __len__(self):
    return len(self.heap)

  def __iter__(self):
    for _, item, _ in self.heap:
      yield item

  def __repr__(self):
    return '[' + ', '.join(repr(c) for c in self.heap) + ']'

  def __str__(self):
    return repr(self)


class RouletteWheel(object):
  """Randomly samples stored objects proportionally to their given weights.

  Stores objects and weights. Acts like a roulette wheel where each object is
  given a slice of the roulette disk proportional to its weight.

  This can be used as a replay buffer where past experiences are sampled
  proportionally to their weights. A good choice of "weight" for reinforcement
  learning is exp(reward / temperature) where temperature -> inf makes the
  distribution more uniform and temperature -> 0 makes the distribution more
  peaky.

  To prevent experiences from being overweighted by appearing in the replay
  buffer multiple times, a "unique mode" is supported where duplicate
  experiences are ignored. In unique mode, weights can be quickly retrieved from
  keys.
  """

  def __init__(self, unique_mode=False, save_file=None):
    """Construct empty RouletteWheel.

    If `save_file` is not None, and the file already exists on disk, whatever
    is in the file will be loaded into this instance. This allows jobs using
    RouletteWheel to resume after preemption.

    Args:
      unique_mode: If True, puts this RouletteWheel into unique mode, where
          objects are added with hashable keys, so that duplicates are ignored.
      save_file: Optional file path to save to. Must be a string containing
          an absolute path to a file, or None. File will be Python pickle
          format.
    """
    self.unique_mode = unique_mode
    self.objects = []
    self.weights = []
    self.partial_sums = []
    if self.unique_mode:
      self.keys_to_weights = {}
    self.save_file = save_file
    self.save_to_disk_buffer = []

    if save_file is not None and tf.gfile.Exists(save_file):
      # Load from disk.
      with tf.gfile.OpenFast(save_file, 'r') as f:
        count = 0
        while 1:
          try:
            obj, weight, key = cPickle.load(f)
          except EOFError:
            break
          else:
            self.add(obj, weight, key)
            count += 1
      logging.info('Loaded %d samples from disk.', count)
      # Clear buffer since these items are already on disk.
      self.save_to_disk_buffer = []

  def __iter__(self):
    return iter(zip(self.objects, self.weights))

  def __len__(self):
    return len(self.objects)

  def is_empty(self):
    """Returns whether there is anything in the roulette wheel."""
    return not self.partial_sums

  @property
  def total_weight(self):
    """Total cumulative weight across all objects."""
    if self.partial_sums:
      return self.partial_sums[-1]
    return 0.0

  def has_key(self, key):
    if self.unique_mode:
      RuntimeError('has_key method can only be called in unique mode.')
    return key in self.keys_to_weights

  def get_weight(self, key):
    if self.unique_mode:
      RuntimeError('get_weight method can only be called in unique mode.')
    return self.keys_to_weights[key]

  def add(self, obj, weight, key=None):
    """Add one object and its weight to the roulette wheel.

    Args:
      obj: Any object to be stored.
      weight: A non-negative float. The given object will be drawn with
          probability proportional to this weight when sampling.
      key: This argument is only used when in unique mode. To allow `obj` to
          be an unhashable type, like list, a separate hashable key is given.
          Each `key` should be unique to each `obj`. `key` is used to check if
          `obj` has been added to the roulette wheel before.

    Returns:
      True if the object was added, False if it was not added due to it being
      a duplicate (this only happens in unique mode).

    Raises:
      ValueError: If `weight` is negative.
      ValueError: If `key` is not given when in unique mode, or if `key` is
          given when not in unique mode.
    """
    if weight < 0:
      raise ValueError('Weight must be non-negative')
    if self.unique_mode:
      if key is None:
        raise ValueError(
            'Hashable key required for objects when unique mode is enabled.')
      if key in self.keys_to_weights:
        # Weight updates are not allowed. Ignore the given value of `weight`.
        return False
      self.keys_to_weights[key] = weight
    elif key is not None:
      raise ValueError(
          'key argument should not be used when unique mode is disabled.')
    self.objects.append(obj)
    self.weights.append(weight)
    self.partial_sums.append(self.total_weight + weight)
    if self.save_file is not None:
      # Record new item in buffer.
      self.save_to_disk_buffer.append((obj, weight, key))
    return True

  def add_many(self, objs, weights, keys=None):
    """Add many object and their weights to the roulette wheel.

    Arguments are the same as the `add` method, except each is a list. Lists
    must all be the same length.

    Args:
      objs: List of objects to be stored.
      weights: List of non-negative floats. See `add` method.
      keys: List of hashable keys. This argument is only used when in unique
          mode. See `add` method.

    Returns:
      Number of objects added. This number will be less than the number of
      objects provided if we are in unique mode and some keys are already
      in the roulette wheel.

    Raises:
      ValueError: If `keys` argument is provided when unique_mode == False, or
          is not provided when unique_mode == True.
      ValueError: If any of the lists are not the same length.
      ValueError: If any of the weights are negative.
    """
    if keys is not None and not self.unique_mode:
      raise ValueError('Not in unique mode. Do not provide keys.')
    elif keys is None and self.unique_mode:
      raise ValueError('In unique mode. You must provide hashable keys.')
    if keys and len(objs) != len(keys):
      raise ValueError('Number of objects does not equal number of keys.')
    if len(objs) != len(weights):
      raise ValueError('Number of objects does not equal number of weights.')
    return sum([self.add(obj, weights[i], key=keys[i] if keys else None)
                for i, obj in enumerate(objs)])

  def sample(self):
    """Spin the roulette wheel.

    Randomly select an object with probability proportional to its weight.

    Returns:
      object: The selected object.
      weight: The weight of the selected object.

    Raises:
      RuntimeError: If the roulette wheel is empty.
    """
    if self.is_empty():
      raise RuntimeError('Trying to sample from empty roulette wheel.')
    spin = random.random() * self.total_weight

    # Binary search.
    i = bisect.bisect_right(self.partial_sums, spin)
    if i == len(self.partial_sums):
      # This should not happen since random.random() will always be strictly
      # less than 1.0, and the last partial sum equals self.total_weight().
      # However it may happen due to rounding error. In that case it is easy to
      # handle this, just select the last object.
      i -= 1

    return self.objects[i], self.weights[i]

  def sample_many(self, count):
    """Spin the roulette wheel `count` times and return the results."""
    if self.is_empty():
      raise RuntimeError('Trying to sample from empty roulette wheel.')
    return [self.sample() for _ in xrange(count)]

  def incremental_save(self, log_info=False):
    """Write new entries to disk.

    This performs an append operation on the `save_file` given in the
    constructor. Any entries added since the last call to `incremental_save`
    will be appended to the file.

    If a new RouletteWheel is constructed with the same `save_file`, all the
    entries written there will be automatically loaded into the instance.
    This is useful when a job resumes after preemption.

    Args:
      log_info: If True, info about this operation will be logged.

    Raises:
      RuntimeError: If `save_file` given in the constructor is None.
    """
    if self.save_file is None:
      raise RuntimeError('Cannot call incremental_save. `save_file` is None.')
    if log_info:
      logging.info('Saving %d new samples to disk.',
                   len(self.save_to_disk_buffer))
    with tf.gfile.OpenFast(self.save_file, 'a') as f:
      for entry in self.save_to_disk_buffer:
        cPickle.dump(entry, f)
    # Clear the buffer.
    self.save_to_disk_buffer = []
