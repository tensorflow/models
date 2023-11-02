# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""File writer functions for dataset preparation, infra validation, and unit tests."""

import io
from typing import Optional, Sequence, Union

import tensorflow as tf, tf_keras


def write_small_dataset(examples: Sequence[Union[tf.train.Example,
                                                 tf.train.SequenceExample]],
                        output_path: str,
                        file_type: str = 'tfrecord') -> None:
  """Writes `examples` to a file at `output_path` with type `file_type`.

  CAVEAT: This function is not recommended for writing large datasets, since it
  will loop through `examples` and perform write operation sequentially.

  Args:
    examples: List of tf.train.Example or tf.train.SequenceExample.
    output_path: Output path for the dataset.
    file_type: A string indicating the file format, could be: 'tfrecord',
      'tfrecords', 'tfrecord_compressed', 'tfrecords_gzip', 'riegeli'. The
      string is case insensitive.
  """
  file_type = file_type.lower()

  if file_type == 'tfrecord' or file_type == 'tfrecords':
    _write_tfrecord(examples, output_path)
  elif file_type == 'tfrecord_compressed' or file_type == 'tfrecords_gzip':
    _write_tfrecord(examples, output_path,
                    tf.io.TFRecordOptions(compression_type='GZIP'))
  elif file_type == 'riegeli':
    _write_riegeli(examples, output_path)
  else:
    raise ValueError(f'Unknown file_type: {file_type}')


def _write_tfrecord(examples: Sequence[Union[tf.train.Example,
                                             tf.train.SequenceExample]],
                    output_path: str,
                    options: Optional[tf.io.TFRecordOptions] = None) -> None:
  """Writes `examples` to a TFRecord file at `output_path`.

  Args:
    examples: A list of tf.train.Example.
    output_path: Output path for the dataset.
    options: Options used for manipulating TFRecord files.
  """
  with tf.io.TFRecordWriter(output_path, options) as writer:
    for example in examples:
      writer.write(example.SerializeToString())


def _write_riegeli(examples: Sequence[Union[tf.train.Example,
                                            tf.train.SequenceExample]],
                   output_path: str) -> None:
  """Writes `examples` to a Riegeli file at `output_path`.

  Args:
    examples: A list of tf.train.Example.
    output_path: Output path for the dataset.
  """
  with io.FileIO(output_path, 'wb') as fileio:
    import riegeli  # pylint: disable=g-import-not-at-top
    with riegeli.RecordWriter(fileio) as writer:
      writer.write_messages(examples)
