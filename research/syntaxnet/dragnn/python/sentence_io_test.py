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

import os
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from dragnn.python import dragnn_ops

from dragnn.python import sentence_io
from syntaxnet import sentence_pb2

FLAGS = tf.app.flags.FLAGS


def setUpModule():
  if not hasattr(FLAGS, 'test_srcdir'):
    FLAGS.test_srcdir = ''
  if not hasattr(FLAGS, 'test_tmpdir'):
    FLAGS.test_tmpdir = tf.test.get_temp_dir()


class ConllSentenceReaderTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # This dataset contains 54 sentences.
    self.filepath = os.path.join(
        FLAGS.test_srcdir,
        'syntaxnet/testdata/mini-training-set')
    self.batch_size = 20

  def assertParseable(self, reader, expected_num, expected_last):
    sentences, last = reader.read()
    self.assertEqual(expected_num, len(sentences))
    self.assertEqual(expected_last, last)
    for s in sentences:
      pb = sentence_pb2.Sentence()
      pb.ParseFromString(s)
      self.assertGreater(len(pb.token), 0)

  def testReadFirstSentence(self):
    reader = sentence_io.ConllSentenceReader(self.filepath, 1)
    sentences, last = reader.read()
    self.assertEqual(1, len(sentences))
    pb = sentence_pb2.Sentence()
    pb.ParseFromString(sentences[0])
    self.assertFalse(last)
    self.assertEqual(
        u'I knew I could do it properly if given the right kind of support .',
        pb.text)

  def testReadFromTextFile(self):
    reader = sentence_io.ConllSentenceReader(self.filepath, self.batch_size)
    self.assertParseable(reader, self.batch_size, False)
    self.assertParseable(reader, self.batch_size, False)
    self.assertParseable(reader, 14, True)
    self.assertParseable(reader, 0, True)
    self.assertParseable(reader, 0, True)

  def testReadAndProjectivize(self):
    reader = sentence_io.ConllSentenceReader(
        self.filepath, self.batch_size, projectivize=True)
    self.assertParseable(reader, self.batch_size, False)
    self.assertParseable(reader, self.batch_size, False)
    self.assertParseable(reader, 14, True)
    self.assertParseable(reader, 0, True)
    self.assertParseable(reader, 0, True)


if __name__ == '__main__':
  googletest.main()
