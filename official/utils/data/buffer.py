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
  except Exception as e:
    tf.logging.error(
        "Failed to cleanup temp file {}. Exception: {}".format(path, e))


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

    self._rows = source_array.shape[0]
    self._row_shape = (1,) + source_array.shape[1:]
    self._values_per_row = int(np.product(self._row_shape))

    self._np_dtype = source_array.dtype
    self._tf_dtype = _TF_DTYPE_MAP[source_array.dtype.name]

    x_view = memoryview(source_array)
    assert x_view.nbytes % self._rows == 0
    self._bytes_per_row = int(x_view.nbytes / self._rows)

  def make_decode_fn(self, multi_row):
    """Construct a decode function to be passed to tf.data.Dataset.map().

    Args:
      multi_row: Boolean of whether the .map() expects multiple rows of data
        in the input tensor (multi_row=True), in which case the decode should
        respect this and return a multi-row decoded tensor. Otherwise map should
        simply return a static sized single row tensor.
    """
    tf_dtype = self._tf_dtype
    row_shape = self.row_shape

    if multi_row:
      def deserialize(input_bytes):
        flat_row = tf.decode_raw(input_bytes, out_type=tf_dtype)
        return tf.reshape(flat_row, (-1,) + row_shape[1:])
    else:
      def deserialize(input_bytes):
        flat_row = tf.decode_raw(input_bytes, out_type=tf_dtype)
        return tf.reshape(flat_row, row_shape[1:])

    return deserialize

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


class _InMemoryArrayBytesView(_ArrayBytesView):
  """Helper class to construct TensorFlow datasets from NumPy data in memory."""

  def __init__(self, source_array, in_place=False):
    """Construct view into array buffer, and enforce ownership if necessary.

    Args:
      source_array: NumPy array to be converted into a dataset.
      in_place: Boolean of whether to use the existing source array buffer
        (true) or create a copy and release the source array (false).
    """
    super(_InMemoryArrayBytesView, self).__init__(source_array)

    if in_place and not source_array.flags.owndata:
      # NumPy arrays are fundamentally a bunch of helper methods sitting on
      # top of a large byte buffer. This allows useful things like two arrays
      # sharing the same memory, or efficient non-copy slices along the major
      # dimension, but for this case it means that another part of the program
      # could manipulate the underlying data in unexpected ways. As a result
      # this case is explicitly not supported.
      raise OSError("Refusing to use memory in place, as input array does not "
                    "own it's own buffer.")

    # TODO(robieta): Not exactly sure how to handle this, so disallowing for now.
    assert not source_array.flags.writebackifcopy

    if not in_place:
      source_array = np.copy(source_array)

    source_array.setflags(write=False)

    # TODO(robieta): check to see if garbage collection of x can be an issue
    self._x_view = memoryview(source_array)

  def to_dataset(self, rows_per_yield=None, decode_procs=2):
    """Create a Dataset from buffer using a generator to encoded bytes.

    Args:
      rows_per_yield: Number of rows worth of bytes to return for each next()
        call. Higher rows_per_yield makes it less likely that python generator
        speed will bottleneck performance.
      decode_procs: Number of parallel decoding map workers called in the
        constructed dataset. Two is often significantly faster than one, but
        very little marginal benefit has been observed above two.
    """
    rows_per_yield = rows_per_yield or _DEFAULT_ROWS_PER_YIELD

    # grab variables for explicit definition rather than including attribute
    # lookups in the function.
    x_view = self._x_view
    rows = self.rows
    num_loops = int(np.ceil(rows / rows_per_yield))

    def row_bytes_generator():
      for i in range(num_loops):
        yield x_view[i * rows_per_yield:(i+1) * rows_per_yield].tobytes()

    dataset = tf.data.Dataset.from_generator(
        row_bytes_generator, output_types=tf.string)
    dataset = dataset.map(self.make_decode_fn(multi_row=True),
                          num_parallel_calls=decode_procs)

    # An alternative to always unbatching is to make rows_per_yield equal to
    # the dataset batch size so that the decode operation automatically produces
    # correctly sized batches. While this can be faster (by ~25%), this dataset
    # is already quite fast and linking the decode and batch operations in such
    # a way makes the resultant dataset less flexible. If there is a significant
    # bottleneck an option can be added use the decode as a batch operation, and
    # make rows_per_yield batch aware.

    # TODO: determine why unbatch() incurs an overhead at dataset creation.
    # dataset = dataset.apply(tf.contrib.data.unbatch())

    # flat_map() and from_tensor_slices are used instead of apply() and
    # unbatch() until the performance difference can be resolved.
    dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    return dataset


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

  def to_dataset(self, decode_procs=2):
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
        filenames=self._buffer_path, record_bytes=self.bytes_per_row)
    dataset = dataset.map(self.make_decode_fn(multi_row=False),
                          num_parallel_calls=decode_procs)

    return dataset


def array_to_dataset(source_array, in_memory=False, chunk_size=None,
                     in_place=None, decode_procs=None, rows_per_yield=None):
  """Helper function to expose view classes."""
  kwarg_dict = {"source_array": source_array}
  if in_memory:
    if in_place is not None:
      kwarg_dict["in_place": in_place]
    return _InMemoryArrayBytesView(**kwarg_dict).to_dataset(
        rows_per_yield=rows_per_yield, decode_procs=decode_procs)
  if chunk_size is not None:
    kwarg_dict["chunk_size"] = chunk_size
  return _FileBackedArrayBytesView(**kwarg_dict).to_dataset(
      decode_procs=decode_procs)
