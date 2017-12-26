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

"""Computes Spearman's rho with respect to human judgements.

Given a set of row (and potentially column) embeddings, this computes Spearman's
rho between the rank ordering of predicted word similarity and human judgements.

Usage:

  wordim.py --embeddings=<binvecs> --vocab=<vocab> eval1.tab eval2.tab ...

Options:

  --embeddings=<filename>: the vectors to test
  --vocab=<filename>: the vocabulary file

Evaluation files are assumed to be tab-separated files with exactly three
columns.  The first two columns contain the words, and the third column contains
the scored human judgement.

"""

from __future__ import print_function
import scipy.stats
import sys
from getopt import GetoptError, getopt

from vecs import Vecs

try:
  opts, args = getopt(sys.argv[1:], '', ['embeddings=', 'vocab='])
except GetoptError as e:
  print(e, file=sys.stderr)
  sys.exit(2)

opt_embeddings = None
opt_vocab = None

for o, a in opts:
  if o == '--embeddings':
    opt_embeddings = a
  if o == '--vocab':
    opt_vocab = a

if not opt_vocab:
  print('please specify a vocabulary file with "--vocab"', file=sys.stderr)
  sys.exit(2)

if not opt_embeddings:
  print('please specify the embeddings with "--embeddings"', file=sys.stderr)
  sys.exit(2)

try:
  vecs = Vecs(opt_vocab, opt_embeddings)
except IOError as e:
  print(e, file=sys.stderr)
  sys.exit(1)


def evaluate(lines):
  acts, preds = [], []

  with open(filename, 'r') as lines:
    for line in lines:
      w1, w2, act = line.strip().split('\t')
      pred = vecs.similarity(w1, w2)
      if pred is None:
        continue

      acts.append(float(act))
      preds.append(pred)

  rho, _ = scipy.stats.spearmanr(acts, preds)
  return rho


for filename in args:
  with open(filename, 'r') as lines:
    print('%0.3f %s' % (evaluate(lines), filename))
