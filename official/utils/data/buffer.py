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
import tempfile
import uuid

import numpy as np
import tensorflow as tf

# I'm sure these already exist somewhere, but for now this is expedient.
_NP_DTYPE_MAP = {
  "uint8": np.uint8,
  "int8": np.int8,
  "int64": np.int64,
  "float32": np.float32,
}

_TF_DTYPE_MAP = {
  "uint8": tf.uint8,
  "int8": tf.int8,
  "int64": tf.int64,
  "float32": tf.float32,
}

# NumPy arrays are written to the underlying file in chunks to prevent
# I/O latency from bottlenecking.
_DEFAULT_CHUNK_SIZE = 1024 ** 2  # 1 MB

# Performance parameter for in-memory dataset creation.
_DEFAULT_ROWS_PER_YIELD = 16


def _cleanup_buffer_file(path):
  try:
    if tf.gfile.Exists(path):
      tf.gfile.Remove(path)
      tf.logging.info("Buffer file {} removed".format(path))
  except Exception as e:
    tf.logging.error(
        "Failed to cleanup temp file {}. Exception: {}".format(path, e))


class _CleanupManager(object):
  """Prevent buildup over the course of execution.

  This module is responsible for creating large blocks of bytes, either as
  buffer files or bytes in memory. Over the course of a long execution, these
  artifacts can accumulate and cause various issues. The purpose of this class
  is to provide an interface for a user to indicate that a view is no longer
  needed, and may be spun down. Otherwise they will be cleaned up at program
  exit.
  """
  def __init__(self):
    self.views = {}

  def register(self, name, view):
    if name in self.views:
      raise ValueError("Name '{}' is already registered. Call `cleanup()` to "
                       "reuse namespaces.".format(name))
    self.views[name] = view

  def cleanup(self, name):
    if name not in self.views:
      return

    view, self.views[name] = self.views[name], None
    view.cleanup()
    del view
    self.views.pop(name)

  def purge(self):
    view_names = list(self.views.keys())
    for i in view_names:
      try:
        self.cleanup(i)
      except Exception as e:
        tf.logging.error("Failed to cleanup view: {} Continuing.".format(e))


_CLEANUP_MANAGER = _CleanupManager()
def cleanup(name):
  return _CLEANUP_MANAGER.cleanup(name=name)

def purge():
  _CLEANUP_MANAGER.purge()
atexit.register(purge)


class _ArrayBytesView(object):
  """Helper class for moving numpy array data into TensorFlow Datasets."""

  def __init__(self, source_array):
    """Check array and store key invariants.

    This class should not "hold onto" a source array; it simply records all
    relevant information. Subclasses may choose to copy the underlying data
    or work directly with the source_array (including locking it), but the base
    class does not assume any claim of ownership of the source array.

    Args:
      source_array: NumPy array to be converted into a dataset.
    """
    assert isinstance(source_array, np.ndarray)
    assert len(source_array.shape) > 0
    assert source_array.shape[0] > 0
    assert source_array.dtype.name in _NP_DTYPE_MAP
    assert source_array.dtype.name in _TF_DTYPE_MAP
    assert source_array.flags.c_contiguous

    self._rows = source_array.shape[0]
    self._row_shape = (1,) + source_array.shape[1:]
    self._values_per_row = int(np.product(self._row_shape))

    self._np_dtype = source_array.dtype
    self._tf_dtype = _TF_DTYPE_MAP[source_array.dtype.name]

    x_view = memoryview(source_array)
    assert x_view.nbytes % self._rows == 0
    self._bytes_per_row = int(x_view.nbytes / self._rows)
    del x_view

  def cleanup(self):
    pass

  @property
  def rows(self):
    return self._rows

  @property
  def row_shape(self):
    return self._row_shape

  @property
  def bytes_per_row(self):
    return self._bytes_per_row

  @property
  def tf_dtype(self):
    return self._tf_dtype


class _FileBackedArrayBytesView(_ArrayBytesView):
  """Write a NumPy array buffer to a file, and then make a dataset from it."""

  def __init__(self, source_array, chunk_size=_DEFAULT_CHUNK_SIZE):
    """Copy array bytes into a temporary buffer file.

    Args:
      source_array: NumPy array to be converted into a dataset.
      chunk_size: data is copied from the array in approximately chunk_size
        segments. This way if there is significant I/O latency (i.e. a
        distributed file system) it is less likely to bottleneck the copy.
    """
    super(_FileBackedArrayBytesView, self).__init__(source_array)

    # TODO(robieta): enable GCS buffer files
    _, self._buffer_path = tempfile.mkstemp()
    atexit.register(_cleanup_buffer_file, path=self._buffer_path)
    self._write_buffer(source_array=source_array, chunk_size=chunk_size)

  def cleanup(self):
    _cleanup_buffer_file(path=self._buffer_path)

  def _write_buffer(self, source_array, chunk_size):
    if tf.gfile.Stat(self._buffer_path).length != 0:
      raise OSError("Buffer file {} exists and is not empty."
                    .format(self._buffer_path))

    rows_per_chunk = int(chunk_size // self.bytes_per_row)
    if rows_per_chunk == 0:
      tf.logging.warning(
          "chunk_size {} is less than the size of a single row ({} bytes). "
          "chunk_size will be rounded up to write one row at a time."
            .format(chunk_size, self.bytes_per_row))

    x_view = memoryview(source_array)
    with tf.gfile.Open(self._buffer_path, "wb") as f:
      for i in range(int(np.ceil(self.rows / rows_per_chunk))):
        chunk = x_view[i * rows_per_chunk:(i+1) * rows_per_chunk].tobytes()
        f.write(chunk)
    del x_view

  def to_dataset(self, decode_procs=2, decode_batch_size=16,
                 extra_map_fn=None, unbatch=True):
    """Create a Dataset from the underlying buffer file.

    Args:
      decode_procs: Number of parallel decoding map workers called in the
        constructed dataset. Two is often significantly faster than one, but
        very little marginal benefit has been observed above two.
    """
    if not tf.gfile.Exists(self._buffer_path):
      raise OSError("Array buffer does not exist.")

    expected_length = self.rows * self.bytes_per_row
    actual_length = tf.gfile.Stat(self._buffer_path).length

    if expected_length != actual_length:
      raise OSError("Array buffer has size {}. Expected size {}."
                    .format(actual_length, expected_length))

    dataset = tf.data.FixedLengthRecordDataset(
        filenames=self._buffer_path, record_bytes=self.bytes_per_row,
        buffer_size=decode_procs * decode_batch_size * self.bytes_per_row * 2)
    dataset = dataset.batch(batch_size=decode_batch_size)

    tf_dtype = self._tf_dtype
    row_shape = self.row_shape

    def deserialize(input_bytes):
      flat_row = tf.decode_raw(input_bytes, out_type=tf_dtype)
      output = tf.reshape(flat_row, (-1,) + row_shape[1:])
      if extra_map_fn:
        output = extra_map_fn(output)
      return output

    dataset = dataset.map(deserialize, num_parallel_calls=decode_procs)

    if unbatch:
      # This is equivalent to:
      #     dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
      # The apply() + unbatch() approach incurs a ~ 2 second overhead when
      # the dataset is defined, but executes significantly faster than the
      # from_tensor_slices() approach, and is therefore generally worth the
      # wait.
      dataset = dataset.apply(tf.contrib.data.unbatch())

    return dataset


def array_to_dataset(source_array, decode_procs=2, decode_batch_size=16,
                     extra_map_fn=None, unbatch=True,
                     namespace=None):
  """Helper function to expose view class."""
  if namespace is None:
    namespace = str(uuid.uuid4().hex)

  view = _FileBackedArrayBytesView(source_array=source_array)
  _CLEANUP_MANAGER.register(name=namespace, view=view)

  return view.to_dataset(decode_procs=decode_procs,
                         decode_batch_size=decode_batch_size,
                         extra_map_fn=extra_map_fn,
                         unbatch=unbatch)
