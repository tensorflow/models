#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Allows the MNIST model to run multi-gpu.

TODO(karmel): When multi-GPU is out of contrib, use core instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import mnist
import tensorflow as tf


def model_fn_with_tower_optimizer(features, labels, mode, params):
  """Wrapper for the model_fn that sets the optimizer as the AdamOptimizer.
  """
  optimizer = mnist.get_optimizer()
  optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
  return mnist.model_fn(features, labels, mode, params, optimizer)


def main(unused_argv):
  replicated_fn = tf.contrib.estimator.replicate_model_fn(
      model_fn_with_tower_optimizer)
  mnist.main_with_model_fn(FLAGS, unused_argv, replicated_fn)


if __name__ == '__main__':
  parser = mnist.MNISTArgParser()
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
