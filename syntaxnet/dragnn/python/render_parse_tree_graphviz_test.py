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
