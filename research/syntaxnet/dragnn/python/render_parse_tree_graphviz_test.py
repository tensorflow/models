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

"""Tests for ....dragnn.python.render_parse_tree_graphviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import googletest
from dragnn.python import render_parse_tree_graphviz
from syntaxnet import sentence_pb2


class RenderParseTreeGraphvizTest(googletest.TestCase):

  def testGiveMeAName(self):
    document = sentence_pb2.Sentence()
    document.token.add(start=0, end=0, word='hi', head=1, label='something')
    document.token.add(start=1, end=1, word='there')
    contents = render_parse_tree_graphviz.parse_tree_graph(document)
    self.assertIn('<polygon', contents)
    self.assertIn('text/html;charset=utf-8;base64', contents)
    self.assertIn('something', contents)
    self.assertIn('hi', contents)
    self.assertIn('there', contents)


if __name__ == '__main__':
  googletest.main()
