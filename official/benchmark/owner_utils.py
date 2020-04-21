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
"""Utils to set Owner annotations on benchmarks.

@owner_utils.Owner('owner_team/user') can be set either at the benchmark class
level / benchmark method level or both.

Runner frameworks can use owner_utils.GetOwner(benchmark_method) to get the
actual owner. Python inheritance for the owner attribute is respected.  (E.g
method level owner takes precedence over class level).

See owner_utils_test for associated tests and more examples.

The decorator can be applied both at the method level and at the class level.

Simple example:
===============

class MLBenchmark:

  @Owner('example_id')
  def benchmark_method_1_gpu(self):
    return True
"""


def Owner(owner_name):
  """Sets the owner attribute on a decorated method or class."""

  def _Wrapper(func_or_class):
    """Sets the benchmark owner attribute."""
    func_or_class.__benchmark__owner__ = owner_name
    return func_or_class

  return _Wrapper


def GetOwner(benchmark_method_or_class):
  """Gets the inherited owner attribute for this benchmark.

  Checks for existence of __benchmark__owner__. If it's not present, looks for
  it in the parent class's attribute list.

  Args:
    benchmark_method_or_class: A benchmark method or class.

  Returns:
    string - the associated owner if present / None.
  """
  if hasattr(benchmark_method_or_class, '__benchmark__owner__'):
    return benchmark_method_or_class.__benchmark__owner__
  elif hasattr(benchmark_method_or_class, '__self__'):
    if hasattr(benchmark_method_or_class.__self__, '__benchmark__owner__'):
      return benchmark_method_or_class.__self__.__benchmark__owner__
  return None
