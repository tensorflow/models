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
"""Construct the spec for the CONLL2017 Parser baseline."""

import tensorflow as tf

from tensorflow.python.platform import gfile

from dragnn.protos import spec_pb2
from dragnn.python import spec_builder

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('spec_file', 'parser_spec.textproto',
                    'Filename to save the spec to.')


def main(unused_argv):
  # Left-to-right, character-based LSTM.
  char2word = spec_builder.ComponentSpecBuilder('char_lstm')
  char2word.set_network_unit(
      name='wrapped_units.LayerNormBasicLSTMNetwork',
      hidden_layer_sizes='256')
  char2word.set_transition_system(name='char-shift-only', left_to_right='true')
  char2word.add_fixed_feature(name='chars', fml='char-input.text-char',
                              embedding_dim=16)

  # Lookahead LSTM reads right-to-left to represent the rightmost context of the
  # words. It gets word embeddings from the char model.
  lookahead = spec_builder.ComponentSpecBuilder('lookahead')
  lookahead.set_network_unit(
      name='wrapped_units.LayerNormBasicLSTMNetwork',
      hidden_layer_sizes='256')
  lookahead.set_transition_system(name='shift-only', left_to_right='false')
  lookahead.add_link(source=char2word, fml='input.last-char-focus',
                     embedding_dim=64)

  # Construct the tagger. This is a simple left-to-right LSTM sequence tagger.
  tagger = spec_builder.ComponentSpecBuilder('tagger')
  tagger.set_network_unit(
      name='wrapped_units.LayerNormBasicLSTMNetwork',
      hidden_layer_sizes='256')
  tagger.set_transition_system(name='tagger')
  tagger.add_token_link(source=lookahead, fml='input.focus', embedding_dim=64)

  # Construct the parser.
  parser = spec_builder.ComponentSpecBuilder('parser')
  parser.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256',
                          layer_norm_hidden='true')
  parser.set_transition_system(name='arc-standard')
  parser.add_token_link(source=lookahead, fml='input.focus', embedding_dim=64)
  parser.add_token_link(
      source=tagger, fml='input.focus stack.focus stack(1).focus',
      embedding_dim=64)

  # Add discrete features of the predicted parse tree so far, like in Parsey
  # McParseface.
  parser.add_fixed_feature(name='labels', embedding_dim=16,
                           fml=' '.join([
                               'stack.child(1).label',
                               'stack.child(1).sibling(-1).label',
                               'stack.child(-1).label',
                               'stack.child(-1).sibling(1).label',
                               'stack(1).child(1).label',
                               'stack(1).child(1).sibling(-1).label',
                               'stack(1).child(-1).label',
                               'stack(1).child(-1).sibling(1).label',
                               'stack.child(2).label',
                               'stack.child(-2).label',
                               'stack(1).child(2).label',
                               'stack(1).child(-2).label']))

  # Recurrent connection for the arc-standard parser. For both tokens on the
  # stack, we connect to the last time step to either SHIFT or REDUCE that
  # token. This allows the parser to build up compositional representations of
  # phrases.
  parser.add_link(
      source=parser,  # recurrent connection
      name='rnn-stack',  # unique identifier
      fml='stack.focus stack(1).focus',  # look for both stack tokens
      source_translator='shift-reduce-step',  # maps token indices -> step
      embedding_dim=64)  # project down to 64 dims

  master_spec = spec_pb2.MasterSpec()
  master_spec.component.extend(
      [char2word.spec, lookahead.spec, tagger.spec, parser.spec])

  with gfile.FastGFile(FLAGS.spec_file, 'w') as f:
    f.write(str(master_spec).encode('utf-8'))

if __name__ == '__main__':
  tf.app.run()
