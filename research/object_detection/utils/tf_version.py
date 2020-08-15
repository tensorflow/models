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
"""Functions to check TensorFlow Version."""

from tensorflow.python import tf2  # pylint: disable=import-outside-toplevel


def is_tf1():
  """Whether current TensorFlow Version is 1.X."""
  return not tf2.enabled()


def is_tf2():
  """Whether current TensorFlow Version is 2.X."""
  return tf2.enabled()
