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

"""Mock objects and related functions for testing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MockBenchmarkLogger(object):
  """This is a mock logger that can be used in dependent tests."""

  def __init__(self):
    self.logged_metric = []

  def log_metric(self, name, value, unit=None, global_step=None,
                 extras=None):
    self.logged_metric.append({
        "name": name,
        "value": float(value),
        "unit": unit,
        "global_step": global_step,
        "extras": extras})
