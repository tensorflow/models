#!/usr/bin/env python
#
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

"""Simple tool for inspecting nearest neighbors and analogies."""

import re
import sys
from getopt import GetoptError, getopt

from vecs import Vecs

try:
  opts, args = getopt(sys.argv[1:], 'v:e:', ['vocab=', 'embeddings='])
except GetoptError, e:
  print >> sys.stderr, e
  sys.exit(2)

opt_vocab = 'vocab.txt'
opt_embeddings = None

for o, a in opts:
  if o in ('-v', '--vocab'):
    opt_vocab = a
  if o in ('-e', '--embeddings'):
    opt_embeddings = a

vecs = Vecs(opt_vocab, opt_embeddings)

while True:
  sys.stdout.write('query> ')
  sys.stdout.flush()

  query = sys.stdin.readline().strip()
  if not query:
    break

  parts = re.split(r'\s+', query)

  if len(parts) == 1:
    res = vecs.neighbors(parts[0])

  elif len(parts) == 3:
    vs = [vecs.lookup(w) for w in parts]
    if any(v is None for v in vs):
      print 'not in vocabulary: %s' % (
          ', '.join(tok for tok, v in zip(parts, vs) if v is None))

      continue

    res = vecs.neighbors(vs[2] - vs[0] + vs[1])

  else:
    print 'use a single word to query neighbors, or three words for analogy'
    continue

  if not res:
    continue

  for word, sim in res[:20]:
    print '%0.4f: %s' % (sim, word)

  print
