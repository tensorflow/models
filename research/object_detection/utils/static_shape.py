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

"""Helper functions to access TensorShape values.

The rank 4 tensor_shape must be of the form [batch_size, height, width, depth].
"""


def get_batch_size(tensor_shape):
  """Returns batch size from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the batch size of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[0].value


def get_height(tensor_shape):
  """Returns height from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the height of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[1].value


def get_width(tensor_shape):
  """Returns width from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the width of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[2].value


def get_depth(tensor_shape):
  """Returns depth from the tensor shape.

  Args:
    tensor_shape: A rank 4 TensorShape.

  Returns:
    An integer representing the depth of the tensor.
  """
  tensor_shape.assert_has_rank(rank=4)
  return tensor_shape[3].value
