# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utility library for picking an appropriate dataset function."""

import functools
from typing import Any, Callable, Type, Union

import tensorflow as tf

PossibleDatasetType = Union[Type[tf.data.Dataset], Callable[[tf.Tensor], Any]]


def pick_dataset_fn(file_type: str) -> PossibleDatasetType:
  if file_type == 'tfrecord':
    return tf.data.TFRecordDataset
  if file_type == 'tfrecord_compressed':
    return functools.partial(tf.data.TFRecordDataset, compression_type='GZIP')
  raise ValueError('Unrecognized file_type: {}'.format(file_type))
