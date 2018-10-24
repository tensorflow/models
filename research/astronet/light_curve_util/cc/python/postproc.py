# Copyright 2018 The TensorFlow Authors.
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

"""Postprocessing utility function for CLIF."""


# CLIF postprocessor for a C++ function with signature:
#   bool MyFunc(input_arg1, ..., *output_arg1, *output_arg2, ..., *error)
#
# If MyFunc returns True, returns (output_arg1, output_arg2, ...)
# If MyFunc returns False, raises ValueError(error).
def ValueErrorOnFalse(ok, *output_args):
  """Raises ValueError if not ok, otherwise returns the output arguments."""
  n_outputs = len(output_args)
  if n_outputs < 2:
    raise ValueError(
        "Expected 2 or more output_args. Got: {}".format(n_outputs))

  if not ok:
    error = output_args[-1]
    raise ValueError(error)

  if n_outputs == 2:
    output = output_args[0]
  else:
    output = output_args[0:-1]

  return output


# CLIF postprocessor for a C++ function with signature:
#   *result MyFactory(input_arg1, ..., *error)
#
# If result is not null, returns result.
# If result is null, raises ValueError(error).
def ValueErrorOnNull(result, error):
  """Raises ValueError(error) if result is None, otherwise returns result."""
  if result is None:
    raise ValueError(error)

  return result
