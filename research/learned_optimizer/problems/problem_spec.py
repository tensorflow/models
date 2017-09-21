# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""Wrapper around a training problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple


class Spec(namedtuple("Spec", "callable args kwargs")):
  """Syntactic sugar for keeping track of a function/class + args."""

  # Since this is an immutable object, we don't need to reserve slots.
  __slots__ = ()

  def build(self):
    """Returns the output of the callable."""
    return self.callable(*self.args, **self.kwargs)
