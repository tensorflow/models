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

"""Provides functions to prefetch tensors to feed into models."""
import tensorflow.compat.v1 as tf


def prefetch(tensor_dict, capacity):
  """Creates a prefetch queue for tensors.

  Creates a FIFO queue to asynchronously enqueue tensor_dicts and returns a
  dequeue op that evaluates to a tensor_dict. This function is useful in
  prefetching preprocessed tensors so that the data is readily available for
  consumers.

  Example input pipeline when you don't need batching:
  ----------------------------------------------------
  key, string_tensor = slim.parallel_reader.parallel_read(...)
  tensor_dict = decoder.decode(string_tensor)
  tensor_dict = preprocessor.preprocess(tensor_dict, ...)
  prefetch_queue = prefetcher.prefetch(tensor_dict, capacity=20)
  tensor_dict = prefetch_queue.dequeue()
  outputs = Model(tensor_dict)
  ...
  ----------------------------------------------------

  For input pipelines with batching, refer to core/batcher.py

  Args:
    tensor_dict: a dictionary of tensors to prefetch.
    capacity: the size of the prefetch queue.

  Returns:
    a FIFO prefetcher queue
  """
  names = list(tensor_dict.keys())
  dtypes = [t.dtype for t in tensor_dict.values()]
  shapes = [t.get_shape() for t in tensor_dict.values()]
  prefetch_queue = tf.PaddingFIFOQueue(capacity, dtypes=dtypes,
                                       shapes=shapes,
                                       names=names,
                                       name='prefetch_queue')
  enqueue_op = prefetch_queue.enqueue(tensor_dict)
  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      prefetch_queue, [enqueue_op]))
  tf.summary.scalar(
      'queue/%s/fraction_of_%d_full' % (prefetch_queue.name, capacity),
      tf.cast(prefetch_queue.size(), dtype=tf.float32) * (1. / capacity))
  return prefetch_queue
