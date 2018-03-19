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
"""Tests for resources."""

from tensorflow.python.platform import googletest

from syntaxnet.util import resources


class ResourcesTest(googletest.TestCase):
  """Testing rig."""

  def testInvalidResource(self):
    for path in [
        'bad/path/to/no/file',
        'syntaxnet/testdata',
        'syntaxnet/testdata/context.pbtxt',
    ]:
      with self.assertRaises(IOError):
        resources.GetSyntaxNetResource(path)
      with self.assertRaises(IOError):
        resources.GetSyntaxNetResourceAsFile(path)

  def testValidResource(self):
    path = 'syntaxnet/testdata/hello.txt'
    self.assertEqual('hello world\n', resources.GetSyntaxNetResource(path))
    with resources.GetSyntaxNetResourceAsFile(path) as resource_file:
      self.assertEqual('hello world\n', resource_file.read())


if __name__ == '__main__':
  googletest.main()
