# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Load binary codes stored as tf.Example in a TFRecord table."""

import tensorflow as tf


def ReadFirstCode(dataset):
  """Read the first example from a binary code RecordIO table."""
  for record in tf.python_io.tf_record_iterator(dataset):
    tf_example = tf.train.Example()
    tf_example.ParseFromString(record)
    break
  return tf_example


def LoadBinaryCode(input_config, batch_size):
  """Load a batch of binary codes from a tf.Example dataset.

  Args:
    input_config: An InputConfig proto containing the input configuration.
    batch_size: Output batch size of examples.

  Returns:
    A batched tensor of binary codes.
  """
  data = input_config.data

  # TODO: Possibly use multiple files (instead of just one).
  file_list = [data]
  filename_queue = tf.train.string_input_producer(file_list,
                                                  capacity=4)
  reader = tf.TFRecordReader()
  _, values = reader.read(filename_queue)

  serialized_example = tf.reshape(values, shape=[1])
  serialized_features = {
      'code_shape': tf.FixedLenFeature([3],
                                       dtype=tf.int64),
      'code': tf.VarLenFeature(tf.float32),
  }
  example = tf.parse_example(serialized_example, serialized_features)

  # 3D shape: height x width x binary_code_depth
  z = example['code_shape']
  code_shape = tf.reshape(tf.cast(z, tf.int32), [3])
  # Un-flatten the binary codes.
  code = tf.reshape(tf.sparse_tensor_to_dense(example['code']), code_shape)

  queue_size = 10
  queue = tf.PaddingFIFOQueue(
      queue_size + 3 * batch_size,
      dtypes=[code.dtype],
      shapes=[[None, None, None]])
  enqueue_op = queue.enqueue([code])
  dequeue_code = queue.dequeue_many(batch_size)
  queue_runner = tf.train.queue_runner.QueueRunner(queue, [enqueue_op])
  tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, queue_runner)

  return dequeue_code
