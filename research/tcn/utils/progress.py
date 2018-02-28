# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""A utility class for reporting processing progress."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime


class Progress(object):
  """A utility class for reporting processing progress."""

  def __init__(self, target_size):
    self.target_size = target_size
    self.current_size = 0
    self.start_time = datetime.datetime.now()

  def Update(self, current_size):
    """Replaces internal current_size with current_size."""
    self.current_size = current_size

  def Add(self, size):
    """Increments internal current_size by size."""
    self.current_size += size

  def __str__(self):
    processed = 1e-5 + self.current_size / float(self.target_size)
    current_time = datetime.datetime.now()
    elapsed = current_time - self.start_time
    eta = datetime.timedelta(
        seconds=elapsed.total_seconds() / processed - elapsed.total_seconds())
    return "%d / %d (elapsed %s eta %s)" % (
        self.current_size, self.target_size,
        str(elapsed).split(".")[0],
        str(eta).split(".")[0])
