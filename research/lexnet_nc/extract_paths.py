#!/usr/bin/env python
# Copyright 2017, 2018 Google, Inc. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys

import spacy
import tensorflow as tf

tf.flags.DEFINE_string('corpus', '', 'Filename of corpus')
tf.flags.DEFINE_string('labeled_pairs', '', 'Filename of labeled pairs')
tf.flags.DEFINE_string('output', '', 'Filename of output file')
FLAGS = tf.flags.FLAGS


def get_path(mod_token, head_token):
  """Returns the path between a modifier token and a head token."""
  # Compute the path from the root to each token.
  mod_ancestors = list(reversed(list(mod_token.ancestors)))
  head_ancestors = list(reversed(list(head_token.ancestors)))

  # If the paths don't start at the same place (odd!) then there is no path at
  # all.
  if (not mod_ancestors or not head_ancestors
      or mod_ancestors[0] != head_ancestors[0]):
    return None

  # Eject elements from the common path until we reach the first differing
  # ancestor.
  ix = 1
  while (ix < len(mod_ancestors) and ix < len(head_ancestors)
         and mod_ancestors[ix] == head_ancestors[ix]):
    ix += 1

  # Construct the path.  TODO: add "satellites", possibly honor sentence
  # ordering between modifier and head rather than just always traversing from
  # the modifier to the head?
  path = ['/'.join(('<X>', mod_token.pos_, mod_token.dep_, '>'))]

  path += ['/'.join((tok.lemma_, tok.pos_, tok.dep_, '>'))
           for tok in reversed(mod_ancestors[ix:])]

  root_token = mod_ancestors[ix - 1]
  path += ['/'.join((root_token.lemma_, root_token.pos_, root_token.dep_, '^'))]

  path += ['/'.join((tok.lemma_, tok.pos_, tok.dep_, '<'))
           for tok in head_ancestors[ix:]]

  path += ['/'.join(('<Y>', head_token.pos_, head_token.dep_, '<'))]

  return '::'.join(path)


def main(_):
  nlp = spacy.load('en_core_web_sm')

  # Grab the set of labeled pairs for which we wish to collect paths.
  with tf.gfile.GFile(FLAGS.labeled_pairs) as fh:
    parts = (l.decode('utf-8').split('\t') for l in fh.read().splitlines())
    labeled_pairs = {(mod, head): rel for mod, head, rel in parts}

  # Create a mapping from each head to the modifiers that are used with it.
  mods_for_head = {
      head: set(hm[1] for hm in head_mods)
      for head, head_mods in itertools.groupby(
          sorted((head, mod) for (mod, head) in labeled_pairs.iterkeys()),
          lambda (head, mod): head)}

  # Collect all the heads that we know about.
  heads = set(mods_for_head.keys())

  # For each sentence that contains a (head, modifier) pair that's in our set,
  # emit the dependency path that connects the pair.
  out_fh = sys.stdout if not FLAGS.output else tf.gfile.GFile(FLAGS.output, 'w')
  in_fh = sys.stdin if not FLAGS.corpus else tf.gfile.GFile(FLAGS.corpus)

  num_paths = 0
  for line, sen in enumerate(in_fh, start=1):
    if line % 100 == 0:
      print('\rProcessing line %d: %d paths' % (line, num_paths),
            end='', file=sys.stderr)

    sen = sen.decode('utf-8').strip()
    doc = nlp(sen)

    for head_token in doc:
      head_text = head_token.text.lower()
      if head_text in heads:
        mods = mods_for_head[head_text]
        for mod_token in doc:
          mod_text = mod_token.text.lower()
          if mod_text in mods:
            path = get_path(mod_token, head_token)
            if path:
              label = labeled_pairs[(mod_text, head_text)]
              line = '\t'.join((mod_text, head_text, label, path, sen))
              print(line.encode('utf-8'), file=out_fh)
              num_paths += 1

  out_fh.close()

if __name__ == '__main__':
  tf.app.run()
