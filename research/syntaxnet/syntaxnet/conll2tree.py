# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A program to generate ASCII trees from conll files."""

import collections
import re

import asciitree
import tensorflow as tf

import syntaxnet.load_parser_ops

from tensorflow.python.platform import tf_logging as logging
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task_context',
                    'syntaxnet/models/parsey_mcparseface/context.pbtxt',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')
flags.DEFINE_string('corpus_name', 'stdin-conll',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')


def to_dict(sentence):
  """Builds a dictionary representing the parse tree of a sentence.

     Note that the suffix "@id" (where 'id' is a number) is appended to each
     element to handle the sentence that has multiple elements with identical
     representation. Those suffix needs to be removed after the asciitree is
     rendered.

  Args:
    sentence: Sentence protocol buffer to represent.
  Returns:
    Dictionary mapping tokens to children.
  """
  token_str = list()
  children = [[] for token in sentence.token]
  root = -1
  for i in range(0, len(sentence.token)):
    token = sentence.token[i]
    token_str.append('%s %s %s @%d' %
                     (token.word, token.tag, token.label, (i+1)))
    if token.head == -1:
      root = i
    else:
      children[token.head].append(i)

  def _get_dict(i):
    d = collections.OrderedDict()
    for c in children[i]:
      d[token_str[c]] = _get_dict(c)
    return d

  tree = collections.OrderedDict()
  tree[token_str[root]] = _get_dict(root)
  return tree


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    src = gen_parser_ops.document_source(batch_size=32,
                                         corpus_name=FLAGS.corpus_name,
                                         task_context=FLAGS.task_context)
    sentence = sentence_pb2.Sentence()
    while True:
      documents, finished = sess.run(src)
      logging.info('Read %d documents', len(documents))
      for d in documents:
        sentence.ParseFromString(d)
        tr = asciitree.LeftAligned()
        d = to_dict(sentence)
        print 'Input: %s' % sentence.text
        print 'Parse:'
        tr_str = tr(d)
        pat = re.compile(r'\s*@\d+$')
        for tr_ln in tr_str.splitlines():
          print pat.sub('', tr_ln)

      if finished:
        break


if __name__ == '__main__':
  tf.app.run()
