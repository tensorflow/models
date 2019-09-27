# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Initializes TPU system for TF 2.0."""

import tensorflow as tf


def tpu_initialize(tpu_address):
  """Initializes TPU for TF 2.0 training.

  Args:
    tpu_address: string, bns address of master TPU worker.

  Returns:
    A TPUClusterResolver.
  """
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=tpu_address)
  if tpu_address not in ('', 'local'):
    tf.config.experimental_connect_to_cluster(cluster_resolver)
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  return cluster_resolver
