# coding=utf-8
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

"""Tests for lexicon_builder."""


# disable=no-name-in-module,unused-import,g-bad-import-order,maybe-no-member
import os.path
import tensorflow as tf

import syntaxnet.load_parser_ops

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging

from syntaxnet import sentence_pb2
from syntaxnet import task_spec_pb2
from syntaxnet.ops import gen_parser_ops

FLAGS = tf.app.flags.FLAGS

CONLL_DOC1 = u'''1 बात _ n NN _ _ _ _ _
2 गलत _ adj JJ _ _ _ _ _
3 हो _ v VM _ _ _ _ _
4 तो _ avy CC _ _ _ _ _
5 गुस्सा _ n NN _ _ _ _ _
6 सेलेब्रिटिज _ n NN _ _ _ _ _
7 को _ psp PSP _ _ _ _ _
8 भी _ avy RP _ _ _ _ _
9 आना _ v VM _ _ _ _ _
10 लाजमी _ adj JJ _ _ _ _ _
11 है _ v VM _ _ _ _ _
12 । _ punc SYM _ _ _ _ _'''

CONLL_DOC2 = u'''1 लेकिन _ avy CC _ _ _ _ _
2 अभिनेत्री _ n NN _ _ _ _ _
3 के _ psp PSP _ _ _ _ _
4 इस _ pn DEM _ _ _ _ _
5 कदम _ n NN _ _ _ _ _
6 से _ psp PSP _ _ _ _ _
7 वहां _ pn PRP _ _ _ _ _
8 रंग _ n NN _ _ _ _ _
9 में _ psp PSP _ _ _ _ _
10 भंग _ adj JJ _ _ _ _ _
11 पड़ _ v VM _ _ _ _ _
12 गया _ v VAUX _ _ _ _ _
13 । _ punc SYM _ _ _ _ _'''

TAGS = ['NN', 'JJ', 'VM', 'CC', 'PSP', 'RP', 'JJ', 'SYM', 'DEM', 'PRP', 'VAUX']

CATEGORIES = ['n', 'adj', 'v', 'avy', 'n', 'psp', 'punc', 'pn']

TOKENIZED_DOCS = u'''बात गलत हो तो गुस्सा सेलेब्रिटिज को भी आना लाजमी है ।
लेकिन अभिनेत्री के इस कदम से वहां रंग में भंग पड़ गया ।
'''

CHARS = u'''अ इ आ क ग ज ट त द न प भ ब य म र ल व ह स ि ा ु ी े ै ो ् ड़ । ं'''

COMMENTS = u'# Line with fake comments.'


class LexiconBuilderTest(test_util.TensorFlowTestCase):

  def setUp(self):
    if not hasattr(FLAGS, 'test_srcdir'):
      FLAGS.test_srcdir = ''
    if not hasattr(FLAGS, 'test_tmpdir'):
      FLAGS.test_tmpdir = tf.test.get_temp_dir()
    self.corpus_file = os.path.join(FLAGS.test_tmpdir, 'documents.conll')
    self.context_file = os.path.join(FLAGS.test_tmpdir, 'context.pbtxt')

  def AddInput(self, name, file_pattern, record_format, context):
    inp = context.input.add()
    inp.name = name
    inp.record_format.append(record_format)
    inp.part.add().file_pattern = file_pattern

  def WriteContext(self, corpus_format):
    context = task_spec_pb2.TaskSpec()
    self.AddInput('documents', self.corpus_file, corpus_format, context)
    for name in ('word-map', 'lcword-map', 'tag-map',
                 'category-map', 'label-map', 'prefix-table',
                 'suffix-table', 'tag-to-category', 'char-map'):
      self.AddInput(name, os.path.join(FLAGS.test_tmpdir, name), '', context)
    logging.info('Writing context to: %s', self.context_file)
    with open(self.context_file, 'w') as f:
      f.write(str(context))

  def ReadNextDocument(self, sess, doc_source):
    doc_str, last = sess.run(doc_source)
    if doc_str:
      doc = sentence_pb2.Sentence()
      doc.ParseFromString(doc_str[0])
    else:
      doc = None
    return doc, last

  def ValidateDocuments(self):
    doc_source = gen_parser_ops.document_source(self.context_file, batch_size=1)
    with self.test_session() as sess:
      logging.info('Reading document1')
      doc, last = self.ReadNextDocument(sess, doc_source)
      self.assertEqual(len(doc.token), 12)
      self.assertEqual(u'लाजमी', doc.token[9].word)
      self.assertFalse(last)
      logging.info('Reading document2')
      doc, last = self.ReadNextDocument(sess, doc_source)
      self.assertEqual(len(doc.token), 13)
      self.assertEqual(u'भंग', doc.token[9].word)
      self.assertFalse(last)
      logging.info('Hitting end of the dataset')
      doc, last = self.ReadNextDocument(sess, doc_source)
      self.assertTrue(doc is None)
      self.assertTrue(last)

  def ValidateTagToCategoryMap(self):
    with file(os.path.join(FLAGS.test_tmpdir, 'tag-to-category'), 'r') as f:
      entries = [line.strip().split('\t') for line in f.readlines()]
    for tag, category in entries:
      self.assertIn(tag, TAGS)
      self.assertIn(category, CATEGORIES)

  def LoadMap(self, map_name):
    loaded_map = {}
    with file(os.path.join(FLAGS.test_tmpdir, map_name), 'r') as f:
      for line in f:
        entries = line.strip().split(' ')
        if len(entries) == 2:
          loaded_map[entries[0]] = entries[1]
    return loaded_map

  def ValidateCharMap(self):
    char_map = self.LoadMap('char-map')
    self.assertEqual(len(char_map), len(CHARS.split(' ')))
    for char in CHARS.split(' '):
      self.assertIn(char.encode('utf-8'), char_map)

  def ValidateWordMap(self):
    word_map = self.LoadMap('word-map')
    for word in filter(None, TOKENIZED_DOCS.replace('\n', ' ').split(' ')):
      self.assertIn(word.encode('utf-8'), word_map)

  def BuildLexicon(self):
    with self.test_session():
      gen_parser_ops.lexicon_builder(task_context=self.context_file).run()

  def testCoNLLFormat(self):
    self.WriteContext('conll-sentence')
    logging.info('Writing conll file to: %s', self.corpus_file)
    with open(self.corpus_file, 'w') as f:
      f.write((CONLL_DOC1 + u'\n\n' + CONLL_DOC2 + u'\n')
              .replace(' ', '\t').encode('utf-8'))
    self.ValidateDocuments()
    self.BuildLexicon()
    self.ValidateTagToCategoryMap()
    self.ValidateCharMap()
    self.ValidateWordMap()

  def testCoNLLFormatExtraNewlinesAndComments(self):
    self.WriteContext('conll-sentence')
    with open(self.corpus_file, 'w') as f:
      f.write((u'\n\n\n' + CONLL_DOC1 + u'\n\n\n' + COMMENTS +
               u'\n\n' + CONLL_DOC2).replace(' ', '\t').encode('utf-8'))
    self.ValidateDocuments()
    self.BuildLexicon()
    self.ValidateTagToCategoryMap()

  def testTokenizedTextFormat(self):
    self.WriteContext('tokenized-text')
    with open(self.corpus_file, 'w') as f:
      f.write(TOKENIZED_DOCS.encode('utf-8'))
    self.ValidateDocuments()
    self.BuildLexicon()

  def testTokenizedTextFormatExtraNewlines(self):
    self.WriteContext('tokenized-text')
    with open(self.corpus_file, 'w') as f:
      f.write((u'\n\n\n' + TOKENIZED_DOCS + u'\n\n\n').encode('utf-8'))
    self.ValidateDocuments()
    self.BuildLexicon()

if __name__ == '__main__':
  googletest.main()
