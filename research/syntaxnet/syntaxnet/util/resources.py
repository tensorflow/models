# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utils for loading resources (data files) from the SyntaxNet source tree.

The resources must be data dependencies of the relevant py_*() build target.

Example usage:

from syntaxnet.util import resources

data_blob = resources.GetSyntaxNetResource(
    'syntaxnet/testdata/context.pbtxt')
"""

import os

# Absolute path to the root directory holding syntaxnet.  Resource paths are
# interpreted relative to this path.

_ROOT_DIR = os.path.dirname(              # .../
    os.path.dirname(                      # .../syntaxnet/
        os.path.dirname(                  # .../syntaxnet/util/
            os.path.abspath(__file__))))  # .../syntaxnet/util/resources.py


def GetSyntaxNetResourceAsFile(path):
  """Returns a resource as an opened read-only file.

  Args:
    path: Relative path to the resource, which must be a Bazel data dependency.

  Returns:
    Opened read-only file pointing to resource data.

  Raises:
    IOError: If the resource cannot be loaded.
  """
  path = os.path.join(_ROOT_DIR, path)
  if os.path.isdir(path):
    raise IOError('Resource "{}" is not a file'.format(path))
  if not os.path.isfile(path):
    raise IOError(
        'Resource "{}" not found; is it a data dependency?'.format(path))
  return open(path, 'rb')


def GetSyntaxNetResource(path):
  """Returns the content of a resource.

  Args:
    path: Relative path to the resource, which must be a Bazel data dependency.

  Returns:
    Raw content of the resource.

  Raises:
    IOError: If the resource cannot be loaded.
  """
  with GetSyntaxNetResourceAsFile(path) as resource_file:
    return resource_file.read()


