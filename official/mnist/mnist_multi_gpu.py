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
from tensorflow.python.client import device_lib


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


def get_reasonable_batch_size(current_size):
  """For multi-gpu, batch-size must be a multiple of the number of
  available GPUs.

  TODO(karmel): This should eventually be handled by replicate_model_fn
  directly. For now, doing the work here.
  """
  devices = _get_local_devices('GPU') or _get_local_devices('CPU')
  num_devices = len(devices)
  remainder = current_size % num_devices
  return current_size - remainder


# TODO(karmel): Replicated from
# tf.contrib.estimator.python.estimator.replicate_model_fn . Should not
# be a copy, but to avoid import problems until this is done by
# replicate_model_fn itself, including here.
def _get_local_devices(device_type):
  local_device_protos = device_lib.list_local_devices()
  return [
      device.name
      for device in local_device_protos
      if device.device_type == device_type
  ]


if __name__ == '__main__':
  parser = mnist.MNISTArgParser()

  # Set default batch size
  batch_size = get_reasonable_batch_size(parser.get_default('batch_size'))
  print('Batch size', batch_size)
  parser.set_defaults(batch_size=batch_size)

  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
