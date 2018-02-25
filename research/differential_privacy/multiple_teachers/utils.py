# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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


def batch_indices(batch_nb, data_length, batch_size):
  """
  This helper function computes a batch start and end index
  :param batch_nb: the batch number
  :param data_length: the total length of the data being parsed by batches
  :param batch_size: the number of inputs in each batch
  :return: pair of (start, end) indices
  """
  # Batch start and end index
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)

  # When there are not enough inputs left, we reuse some to complete the batch
  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift

  return start, end
