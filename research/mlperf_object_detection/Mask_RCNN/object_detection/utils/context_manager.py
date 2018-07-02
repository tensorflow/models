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
"""Python context management helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class IdentityContextManager(object):
  """Returns an identity context manager that does nothing.

  This is helpful in setting up conditional `with` statement as below:

  with slim.arg_scope(x) if use_slim_scope else IdentityContextManager():
    do_stuff()

  """

  def __enter__(self):
    return None

  def __exit__(self, exec_type, exec_value, traceback):
    del exec_type
    del exec_value
    del traceback
    return False

