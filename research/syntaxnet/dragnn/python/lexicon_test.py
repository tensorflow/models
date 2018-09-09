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
"""Tests for SyntaxNet lexicon."""

import os
import os.path

import tensorflow as tf

from google.protobuf import text_format

from dragnn.python import lexicon

# Imported for FLAGS.tf_master, which is used in the lexicon module.

from syntaxnet import parser_trainer
from syntaxnet import task_spec_pb2
from syntaxnet import test_flags


_EXPECTED_CONTEXT = r"""
input { name: "word-map" Part { file_pattern: "/tmp/word-map" } }
input { name: "tag-map" Part { file_pattern: "/tmp/tag-map" } }
input { name: "tag-to-category" Part { file_pattern: "/tmp/tag-to-category" } }
input { name: "lcword-map" Part { file_pattern: "/tmp/lcword-map" } }
input { name: "category-map" Part { file_pattern: "/tmp/category-map" } }
input { name: "char-map" Part { file_pattern: "/tmp/char-map" } }
input { name: "char-ngram-map" Part { file_pattern: "/tmp/char-ngram-map" } }
input { name: "label-map" Part { file_pattern: "/tmp/label-map" } }
input { name: "prefix-table" Part { file_pattern: "/tmp/prefix-table" } }
input { name: "suffix-table" Part { file_pattern: "/tmp/suffix-table" } }
input { name: "known-word-map" Part { file_pattern: "/tmp/known-word-map" } }
"""


class LexiconTest(tf.test.TestCase):

  def testCreateLexiconContext(self):
    expected_context = task_spec_pb2.TaskSpec()
    text_format.Parse(_EXPECTED_CONTEXT, expected_context)
    self.assertProtoEquals(
        lexicon.create_lexicon_context('/tmp'), expected_context)

  def testBuildLexicon(self):
    empty_input_path = os.path.join(test_flags.temp_dir(), 'empty-input')
    lexicon_output_path = os.path.join(test_flags.temp_dir(), 'lexicon-output')

    with open(empty_input_path, 'w'):
      pass

    # The directory may already exist when running locally multiple times.
    if not os.path.exists(lexicon_output_path):
      os.mkdir(lexicon_output_path)

    # Just make sure this doesn't crash; the lexicon builder op is already
    # exercised in its own unit test.
    lexicon.build_lexicon(lexicon_output_path, empty_input_path)


if __name__ == '__main__':
  tf.test.main()
