# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Provides functions to batch a dictionary of input tensors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import range
import tensorflow.compat.v1 as tf

from object_detection.core import prefetcher

rt_shape_str = '_runtime_shapes'


class BatchQueue(object):
  """BatchQueue class.

  This class creates a batch queue to asynchronously enqueue tensors_dict.
  It also adds a FIFO prefetcher so that the batches are readily available
  for the consumers.  Dequeue ops for a BatchQueue object can be created via
  the Dequeue method which evaluates to a batch of tensor_dict.

  Example input pipeline with batching:
  ------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  batch_queue = batcher.BatchQueue(tensor_dict,
                                   batch_size=32,
                                   batch_queue_capacity=2000,
                                   num_batch_queue_threads=8,
                                   prefetch_queue_capacity=20)
  tensor_dict = batch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  -----------------------------------

  Notes:
  -----
  This class batches tensors of unequal sizes by zero padding and unpadding
  them after generating a batch. This can be computationally expensive when
  batching tensors (such as images) that are of vastly different sizes. So it is
  recommended that the shapes of such tensors be fully defined in tensor_dict
  while other lightweight tensors such as bounding box corners and class labels
  can be of varying sizes. Use either crop or resize operations to fully define
  the shape of an image in tensor_dict.

  It is also recommended to perform any preprocessing operations on tensors
  before passing to BatchQueue and subsequently calling the Dequeue method.

  Another caveat is that this class does not read the last batch if it is not
  full. The current implementation makes it hard to support that use case. So,
  for evaluation, when it is critical to run all the examples through your
  network use the input pipeline example mentioned in core/prefetcher.py.
  """

  def __init__(self, tensor_dict, batch_size, batch_queue_capacity,
               num_batch_queue_threads, prefetch_queue_capacity):
    """Constructs a batch queue holding tensor_dict.

    Args:
      tensor_dict: dictionary of tensors to batch.
      batch_size: batch size.
      batch_queue_capacity: max capacity of the queue from which the tensors are
        batched.
      num_batch_queue_threads: number of threads to use for batching.
      prefetch_queue_capacity: max capacity of the queue used to prefetch
        assembled batches.
    """
    # Remember static shapes to set shapes of batched tensors.
    static_shapes = collections.OrderedDict(
        {key: tensor.get_shape() for key, tensor in tensor_dict.items()})
    # Remember runtime shapes to unpad tensors after batching.
    runtime_shapes = collections.OrderedDict(
        {(key + rt_shape_str): tf.shape(tensor)
         for key, tensor in tensor_dict.items()})

    all_tensors = tensor_dict
    all_tensors.update(runtime_shapes)
    batched_tensors = tf.train.batch(
        all_tensors,
        capacity=batch_queue_capacity,
        batch_size=batch_size,
        dynamic_pad=True,
        num_threads=num_batch_queue_threads)

    self._queue = prefetcher.prefetch(batched_tensors,
                                      prefetch_queue_capacity)
    self._static_shapes = static_shapes
    self._batch_size = batch_size

  def dequeue(self):
    """Dequeues a batch of tensor_dict from the BatchQueue.

    TODO: use allow_smaller_final_batch to allow running over the whole eval set

    Returns:
      A list of tensor_dicts of the requested batch_size.
    """
    batched_tensors = self._queue.dequeue()
    # Separate input tensors from tensors containing their runtime shapes.
    tensors = {}
    shapes = {}
    for key, batched_tensor in batched_tensors.items():
      unbatched_tensor_list = tf.unstack(batched_tensor)
      for i, unbatched_tensor in enumerate(unbatched_tensor_list):
        if rt_shape_str in key:
          shapes[(key[:-len(rt_shape_str)], i)] = unbatched_tensor
        else:
          tensors[(key, i)] = unbatched_tensor

    # Undo that padding using shapes and create a list of size `batch_size` that
    # contains tensor dictionaries.
    tensor_dict_list = []
    batch_size = self._batch_size
    for batch_id in range(batch_size):
      tensor_dict = {}
      for key in self._static_shapes:
        tensor_dict[key] = tf.slice(tensors[(key, batch_id)],
                                    tf.zeros_like(shapes[(key, batch_id)]),
                                    shapes[(key, batch_id)])
        tensor_dict[key].set_shape(self._static_shapes[key])
      tensor_dict_list.append(tensor_dict)

    return tensor_dict_list
