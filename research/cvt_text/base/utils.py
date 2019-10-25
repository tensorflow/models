# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Various utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import sys
import tensorflow as tf


class Memoize(object):
  def __init__(self, f):
    self.f = f
    self.cache = {}

  def __call__(self, *args):
    if args not in self.cache:
      self.cache[args] = self.f(*args)
    return self.cache[args]


def load_cpickle(path, memoized=True):
  return _load_cpickle_memoize(path) if memoized else _load_cpickle(path)


def _load_cpickle(path):
  with tf.gfile.GFile(path, 'r') as f:
    return cPickle.load(f)


@Memoize
def _load_cpickle_memoize(path):
  return _load_cpickle(path)


def write_cpickle(o, path):
  tf.gfile.MakeDirs(path.rsplit('/', 1)[0])
  with tf.gfile.GFile(path, 'w') as f:
    cPickle.dump(o, f, -1)


def log(*args):
  msg = ' '.join(map(str, args))
  sys.stdout.write(msg + '\n')
  sys.stdout.flush()


def heading(*args):
  log()
  log(80 * '=')
  log(*args)
  log(80 * '=')
