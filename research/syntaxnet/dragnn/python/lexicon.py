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
"""SyntaxNet lexicon utils."""

import os.path


import tensorflow as tf

from syntaxnet import task_spec_pb2
from syntaxnet.ops import gen_parser_ops


def create_lexicon_context(path):
  """Construct a SyntaxNet TaskContext file for standard lexical resources."""
  context = task_spec_pb2.TaskSpec()
  for name in [
      'word-map', 'tag-map', 'tag-to-category', 'lcword-map', 'category-map',
      'char-map', 'char-ngram-map', 'label-map', 'prefix-table', 'suffix-table',
      'known-word-map'
  ]:
    context.input.add(name=name).part.add(file_pattern=os.path.join(path, name))
  return context


def build_lexicon(output_path,
                  training_corpus_path,
                  tf_master='',
                  training_corpus_format='conll-sentence',
                  morph_to_pos=False,
                  **kwargs):
  """Constructs a SyntaxNet lexicon at the given path.

  Args:
    output_path: Location to construct the lexicon.
    training_corpus_path: Path to CONLL formatted training data.
    tf_master: TensorFlow master executor (string, defaults to '' to use the
      local instance).
    training_corpus_format: Format of the training corpus (defaults to CONLL;
      search for REGISTER_SYNTAXNET_DOCUMENT_FORMAT for other formats).
    morph_to_pos: Whether to serialize morph attributes to the tag field,
      combined with category and fine POS tag.
    **kwargs: Forwarded to the LexiconBuilder op.
  """
  context = create_lexicon_context(output_path)
  if morph_to_pos:
    context.parameter.add(name='join_category_to_pos', value='true')
    context.parameter.add(name='add_pos_as_attribute', value='true')
    context.parameter.add(name='serialize_morph_to_pos', value='true')

  # Add the training data to the context.
  resource = context.input.add()
  resource.name = 'corpus'
  resource.record_format.extend([training_corpus_format])
  part = resource.part.add()
  part.file_pattern = training_corpus_path

  # Run the lexicon builder op.
  with tf.Session(tf_master) as sess:
    sess.run(
        gen_parser_ops.lexicon_builder(
            task_context_str=str(context), corpus_name='corpus', **kwargs))
