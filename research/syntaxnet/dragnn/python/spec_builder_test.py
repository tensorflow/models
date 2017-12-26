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
"""Tests for the DRAGNN spec builder."""

import os.path
import tempfile

import tensorflow as tf

from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import spec_builder

# Imported for FLAGS.tf_master, which is used in the lexicon module.

from syntaxnet import parser_trainer

FLAGS = tf.app.flags.FLAGS


def setUpModule():
  if not hasattr(FLAGS, 'test_srcdir'):
    FLAGS.test_srcdir = ''
  if not hasattr(FLAGS, 'test_tmpdir'):
    FLAGS.test_tmpdir = tf.test.get_temp_dir()


class SpecBuilderTest(tf.test.TestCase):

  def assertSpecEqual(self, expected_spec_text, spec):
    expected_spec = spec_pb2.ComponentSpec()
    text_format.Parse(expected_spec_text, expected_spec)
    self.assertProtoEquals(expected_spec, spec)

  def testComponentSpecBuilderEmpty(self):
    builder = spec_builder.ComponentSpecBuilder('test')
    self.assertSpecEqual(r"""
name: "test"
backend { registered_name: "SyntaxNetComponent" }
component_builder { registered_name: "DynamicComponentBuilder" }
        """, builder.spec)

  def testComponentSpecBuilderFixedFeature(self):
    builder = spec_builder.ComponentSpecBuilder('test')
    builder.set_network_unit('FeedForwardNetwork', hidden_layer_sizes='64,64')
    builder.set_transition_system('shift-only')
    builder.add_fixed_feature(name='words', fml='input.word', embedding_dim=16)
    self.assertSpecEqual(r"""
name: "test"
fixed_feature { name: "words" fml: "input.word" embedding_dim: 16 }
backend { registered_name: "SyntaxNetComponent" }
component_builder { registered_name: "DynamicComponentBuilder" }
network_unit { registered_name: "FeedForwardNetwork"
               parameters { key: "hidden_layer_sizes" value: "64,64" } }
transition_system { registered_name: "shift-only" }
        """, builder.spec)

  def testComponentSpecBuilderLinkedFeature(self):
    builder1 = spec_builder.ComponentSpecBuilder('test1')
    builder1.set_transition_system('shift-only')
    builder1.add_fixed_feature(name='words', fml='input.word', embedding_dim=16)
    builder2 = spec_builder.ComponentSpecBuilder('test2')
    builder2.set_network_unit('IdentityNetwork')
    builder2.set_transition_system('tagger')
    builder2.add_token_link(
        source=builder1,
        source_layer='words',
        fml='input.focus',
        embedding_dim=-1)
    self.assertSpecEqual(r"""
name: "test2"
linked_feature { name: "test1" source_component: "test1" source_layer: "words"
                 source_translator: "identity" fml: "input.focus"
                 embedding_dim: -1 }
backend { registered_name: "SyntaxNetComponent" }
component_builder { registered_name: "DynamicComponentBuilder" }
network_unit { registered_name: "IdentityNetwork" }
transition_system { registered_name: "tagger" }
        """, builder2.spec)

  def testFillsTaggerTransitions(self):
    lexicon_dir = tempfile.mkdtemp()

    def write_lines(filename, lines):
      with open(os.path.join(lexicon_dir, filename), 'w') as f:
        f.write(''.join('{}\n'.format(line) for line in lines))

    # Label map is required, even though it isn't used
    write_lines('label-map', ['0'])
    write_lines('word-map', ['2', 'miranda 1', 'rights 1'])
    write_lines('tag-map', ['2', 'NN 1', 'NNP 1'])
    write_lines('tag-to-category', ['NN\tNOUN', 'NNP\tNOUN'])

    tagger = spec_builder.ComponentSpecBuilder('tagger')
    tagger.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256')
    tagger.set_transition_system(name='tagger')
    tagger.add_fixed_feature(name='words', fml='input.word', embedding_dim=64)
    tagger.add_rnn_link(embedding_dim=-1)
    tagger.fill_from_resources(lexicon_dir)

    fixed_feature, = tagger.spec.fixed_feature
    linked_feature, = tagger.spec.linked_feature
    self.assertEqual(fixed_feature.vocabulary_size, 5)
    self.assertEqual(fixed_feature.size, 1)
    self.assertEqual(fixed_feature.size, 1)
    self.assertEqual(linked_feature.size, 1)
    self.assertEqual(tagger.spec.num_actions, 2)


if __name__ == '__main__':
  tf.test.main()
