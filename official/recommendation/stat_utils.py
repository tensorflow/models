# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Statistics utility functions of NCF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def random_int32():
  return np.random.randint(low=0, high=np.iinfo(np.int32).max, dtype=np.int32)


def permutation(args):
  """Fork safe permutation function.

  This function can be called within a multiprocessing worker and give
  appropriately random results.

  Args:
    args: A size two tuple that will unpacked into the size of the permutation
      and the random seed. This form is used because starmap is not universally
      available.
  Returns:
    A NumPy array containing a random permutation.
  """
  x, seed = args

  # If seed is None NumPy will seed randomly.
  state = np.random.RandomState(seed=seed)  # pylint: disable=no-member
  output = np.arange(x, dtype=np.int32)
  state.shuffle(output)
  return output


def very_slightly_biased_randint(max_val_vector):
  sample_dtype = np.uint64
  out_dtype = max_val_vector.dtype
  samples = np.random.randint(
      low=0,
      high=np.iinfo(sample_dtype).max,
      size=max_val_vector.shape,
      dtype=sample_dtype)
  return np.mod(samples, max_val_vector.astype(sample_dtype)).astype(out_dtype)


def mask_duplicates(x, axis=1):  # type: (np.ndarray, int) -> np.ndarray
  """Identify duplicates from sampling with replacement.

  Args:
    x: A 2D NumPy array of samples
    axis: The axis along which to de-dupe.

  Returns:
    A NumPy array with the same shape as x with one if an element appeared
    previously along axis 1, else zero.
  """
  if axis != 1:
    raise NotImplementedError

  x_sort_ind = np.argsort(x, axis=1, kind="mergesort")
  sorted_x = x[np.arange(x.shape[0])[:, np.newaxis], x_sort_ind]

  # compute the indices needed to map values back to their original position.
  inv_x_sort_ind = np.argsort(x_sort_ind, axis=1, kind="mergesort")

  # Compute the difference of adjacent sorted elements.
  diffs = sorted_x[:, :-1] - sorted_x[:, 1:]

  # We are only interested in whether an element is zero. Therefore left padding
  # with ones to restore the original shape is sufficient.
  diffs = np.concatenate(
      [np.ones((diffs.shape[0], 1), dtype=diffs.dtype), diffs], axis=1)

  # Duplicate values will have a difference of zero. By definition the first
  # element is never a duplicate.
  return np.where(diffs[np.arange(x.shape[0])[:, np.newaxis], inv_x_sort_ind],
                  0, 1)
