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
"""Converts a text embedding file into a binary format for quicker loading."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string('input', '', 'text file containing embeddings')
tf.flags.DEFINE_string('output_vocab', '', 'output file for vocabulary')
tf.flags.DEFINE_string('output_npy', '', 'output file for binary')
FLAGS = tf.flags.FLAGS

def main(_):
  vecs = []
  vocab = []
  with tf.gfile.GFile(FLAGS.input) as fh:
    for line in fh:
      parts = line.strip().split()
      vocab.append(parts[0])
      vecs.append([float(x) for x in parts[1:]])

  with tf.gfile.GFile(FLAGS.output_vocab, 'w') as fh:
    fh.write('\n'.join(vocab))
    fh.write('\n')

  vecs = np.array(vecs, dtype=np.float32)
  np.save(FLAGS.output_npy, vecs, allow_pickle=False)


if __name__ == '__main__':
  tf.app.run()
