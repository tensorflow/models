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

import atexit
import multiprocessing
import timeit

import numpy as np
import pandas as pd
import tempfile

import tensorflow as tf

from official.utils.data import buffer


_BATCH_SIZE = 512
_NUM_BATCHES = 5000
_NUM_PTS = _BATCH_SIZE * _NUM_BATCHES
_NUM_COLS = 6
_DUMMY_KEY = "a"  # TFRecords has to include keys, so using a one letter ascii
                  # key helps performance.

_FEATURE_MAP = {
  _DUMMY_KEY: tf.FixedLenFeature([_NUM_COLS], dtype=tf.int64),
}


def make_array():
  """Construct data arrays for testing.

  This function makes the same array from scratch. This way experiments are
  guaranteed not to interact.
  """
  np.random.seed(1)
  shape = (_NUM_PTS, _NUM_COLS)
  return np.random.randint(0, 2 ** 8 - 1, shape, dtype=np.uint8)


class TimeSections(object):
  def __init__(self):
    self.section_times = {}
    self.total_time = 0
    self.current_section = None
    self.start_time = -1

  def name_section(self, name):
    self.current_section = name

  def __enter__(self):
    self.start_time = timeit.default_timer()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    run_time = timeit.default_timer() - self.start_time
    if self.current_section is None:
      self.current_section = "section_{}".format(len(self.section_times))
    assert self.current_section not in self.section_times
    self.section_times[self.current_section] = run_time
    self.current_section = None


def consume_dataset(dataset, source_array, num_parallel=16):
  # type: (tf.data.Dataset) -> None
  dataset = dataset.batch(_BATCH_SIZE)
  dataset = dataset.prefetch(1)

  g = tf.Graph()
  # print(len(str(g.as_graph_def()).encode("utf-8")))
  with g.as_default():
    row = dataset.make_one_shot_iterator().get_next()
  # print(len(str(g.as_graph_def()).encode("utf-8")))

  with tf.Session(graph=g).as_default() as sess:
    for i in range(int(_NUM_BATCHES // num_parallel)):
      results = sess.run([row for _ in range(num_parallel)])
      if i == 0:
        # Sanity check
        assert np.allclose(results[0], source_array[:_BATCH_SIZE])


def _deserialize(example_byte_tensor):
  return tf.parse_single_example(example_byte_tensor, _FEATURE_MAP)[_DUMMY_KEY]


def make_simple_tfrecord_experiment():
  """Convert NumPy array to TFRecords using a single thread.

  The task of serializing to TFRecords is naively parallelizable, but not
  trivial to implement, requiring various sharding and multiprocessing
  machinery. This is intended as a baseline of what can be accomplished fairly
  easily.
  """
  data = make_array()
  _, buffer_path = tempfile.mkstemp()
  atexit.register(buffer._cleanup_buffer_file, path=buffer_path)
  section_timer = TimeSections()

  with section_timer as t:
    t.name_section("serialize")
    with tf.python_io.TFRecordWriter(buffer_path) as writer:
      for row in data:
        feature = tf.train.Feature(int64_list=tf.train.Int64List(value=row))
        example = tf.train.Example(features=tf.train.Features(
            feature={_DUMMY_KEY: feature}))
        example_bytes = example.SerializeToString()
        writer.write(example_bytes)

  with section_timer as t:
    t.name_section("define_dataset")
    dataset = tf.data.TFRecordDataset(filenames=buffer_path)
    dataset = dataset.map(_deserialize,
                          num_parallel_calls=multiprocessing.cpu_count())

  with section_timer as t:
    t.name_section("consume")
    consume_dataset(dataset, source_array=data)

  return section_timer.section_times


def _serialize_shard(shard):
  # type: (np.ndarray) -> list
  output = []
  for row in shard:
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=row))
    example = tf.train.Example(features=tf.train.Features(
        feature={_DUMMY_KEY: feature}))
    example_bytes = example.SerializeToString()
    output.append(example_bytes)
  return output


def make_tfrecord_parallel_experiment(num_cores=4):
  data = make_array()
  section_timer = TimeSections()

  boundaries = np.linspace(0, _NUM_PTS, num_cores+1).astype("int")
  shards = []
  for i in range(num_cores):
    shards.append(data[boundaries[i]:boundaries[i+1]])

  with section_timer as t:
    t.name_section("serialize")
    with multiprocessing.Pool(processes=num_cores) as pool:
      encoded_shards = pool.map(_serialize_shard, shards)

    _, buffer_path = tempfile.mkstemp()
    atexit.register(buffer._cleanup_buffer_file, path=buffer_path)

    with tf.python_io.TFRecordWriter(buffer_path) as writer:
      for section in encoded_shards:
        for example in section:
          writer.write(example)


  with section_timer as t:
    t.name_section("define_dataset")
    dataset = tf.data.TFRecordDataset(filenames=buffer_path)
    dataset = dataset.map(_deserialize,
                          num_parallel_calls=multiprocessing.cpu_count())

  with section_timer as t:
    t.name_section("consume")
    consume_dataset(dataset, source_array=data)

  return section_timer.section_times


def make_in_memory_buffer_experiment(in_place=False):
  data = make_array()
  section_timer = TimeSections()

  with section_timer as t:
    t.name_section("serialize")
    array_view = buffer._InMemoryArrayBytesView(
        source_array=data, in_place=in_place)

  with section_timer as t:
    t.name_section("define_dataset")
    dataset = array_view.to_dataset(decode_procs=8, rows_per_yield=128)

  with section_timer as t:
    t.name_section("consume")
    consume_dataset(dataset, source_array=data)

  return section_timer.section_times


def make_file_buffer_experiment():
  data = make_array()
  section_timer = TimeSections()

  with section_timer as t:
    t.name_section("serialize")
    array_view = buffer._FileBackedArrayBytesView(source_array=data)

  with section_timer as t:
    t.name_section("define_dataset")
    dataset = array_view.to_dataset(decode_procs=8, decode_batch_size=32)

  with section_timer as t:
    t.name_section("consume")
    consume_dataset(dataset, source_array=data)

  return section_timer.section_times

def pretty_print(timings, name):
  print(name)
  print("Total: {:.2f}".format(sum([i for i in timings.values()])))
  print("  Serialize:   {:.2f}".format(timings["serialize"]))
  print("  Instantiate: {:.2f}".format(timings["define_dataset"]))
  print("  Consume:     {:.2f}".format(timings["consume"]))
  print()


def main():
  # pretty_print(make_simple_tfrecord_experiment(), "Simple TFRecords.")
  num_cores=min([multiprocessing.cpu_count(), 8])
  pretty_print(make_tfrecord_parallel_experiment(num_cores=num_cores),
               "Parallel TFRecords (num_cores: {})".format(num_cores))
  pretty_print(make_in_memory_buffer_experiment(False), "NumPy Buffer (in-memory, copy)")
  pretty_print(make_in_memory_buffer_experiment(True), "NumPy Buffer (in-memory, in-place)")
  pretty_print(make_file_buffer_experiment(), "NumPy Buffer (file backed)")


if __name__ == "__main__":
  main()
