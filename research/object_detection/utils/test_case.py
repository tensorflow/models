# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A convenience wrapper around tf.test.TestCase to test with TPU, TF1, TF2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import zip
import tensorflow as tf
from tensorflow.python import tf2  # pylint: disable=import-outside-toplevel
if not tf2.enabled():
  from tensorflow.contrib import tpu as contrib_tpu  # pylint: disable=g-import-not-at-top, line-too-long

flags = tf.app.flags

flags.DEFINE_bool('tpu_test', False, 'Deprecated Flag.')
FLAGS = flags.FLAGS


class TestCase(tf.test.TestCase):
  """Base Test class to handle execution under {TF1.X, TF2.X} x {TPU, CPU}.

  This class determines the TF version and availability of TPUs to set up
  tests appropriately.
  """

  def maybe_extract_single_output(self, outputs):
    if isinstance(outputs, list) or isinstance(outputs, tuple):
      if isinstance(outputs[0], tf.Tensor):
        outputs_np = [output.numpy() for output in outputs]
      else:
        outputs_np = outputs
      if len(outputs_np) == 1:
        return outputs_np[0]
      else:
        return outputs_np
    else:
      if isinstance(outputs, tf.Tensor):
        return outputs.numpy()
      else:
        return outputs

  def has_tpu(self):
    """Returns whether there are any logical TPU devices."""
    return bool(tf.config.experimental.list_logical_devices(device_type='TPU'))

  def is_tf2(self):
    """Returns whether TF2 is enabled."""
    return tf2.enabled()

  def execute_tpu_tf1(self, compute_fn, inputs):
    """Executes compute_fn on TPU with Tensorflow 1.X.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single numpy array.
    """
    with self.test_session(graph=tf.Graph()) as sess:
      placeholders = [tf.placeholder_with_default(v, v.shape) for v in inputs]
      def wrap_graph_fn(*args, **kwargs):
        results = compute_fn(*args, **kwargs)
        if (not (isinstance(results, dict) or isinstance(results, tf.Tensor))
            and hasattr(results, '__iter__')):
          results = list(results)
        return results
      tpu_computation = contrib_tpu.rewrite(wrap_graph_fn, placeholders)
      sess.run(contrib_tpu.initialize_system())
      sess.run([tf.global_variables_initializer(), tf.tables_initializer(),
                tf.local_variables_initializer()])
      materialized_results = sess.run(tpu_computation,
                                      feed_dict=dict(zip(placeholders, inputs)))
      sess.run(contrib_tpu.shutdown_system())
    return self.maybe_extract_single_output(materialized_results)

  def execute_tpu_tf2(self, compute_fn, inputs):
    """Executes compute_fn on TPU with Tensorflow 2.X.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single numpy array.
    """
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology, num_replicas=1)
    strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=device_assignment)

    @tf.function
    def run():
      tf_inputs = [tf.constant(input_t) for input_t in inputs]
      return strategy.run(compute_fn, args=tf_inputs)
    outputs = run()
    tf.tpu.experimental.shutdown_tpu_system()
    return self.maybe_extract_single_output(outputs)

  def execute_cpu_tf1(self, compute_fn, inputs):
    """Executes compute_fn on CPU with Tensorflow 1.X.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single numpy array.
    """
    if self.is_tf2():
      raise ValueError('Required version Tenforflow 1.X is not available.')
    with self.test_session(graph=tf.Graph()) as sess:
      placeholders = [tf.placeholder_with_default(v, v.shape) for v in inputs]
      results = compute_fn(*placeholders)
      if (not (isinstance(results, dict) or isinstance(results, tf.Tensor)) and
          hasattr(results, '__iter__')):
        results = list(results)
      sess.run([tf.global_variables_initializer(), tf.tables_initializer(),
                tf.local_variables_initializer()])
      materialized_results = sess.run(results, feed_dict=dict(zip(placeholders,
                                                                  inputs)))
    return self.maybe_extract_single_output(materialized_results)

  def execute_cpu_tf2(self, compute_fn, inputs):
    """Executes compute_fn on CPU with Tensorflow 2.X.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single numpy array.
    """
    if not self.is_tf2():
      raise ValueError('Required version TensorFlow 2.0 is not available.')
    @tf.function
    def run():
      tf_inputs = [tf.constant(input_t) for input_t in inputs]
      return compute_fn(*tf_inputs)
    return self.maybe_extract_single_output(run())

  def execute_cpu(self, compute_fn, inputs):
    """Executes compute_fn on CPU.

    Depending on the underlying TensorFlow installation (build deps) runs in
    either TF 1.X or TF 2.X style.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single tensor.
    """
    if self.is_tf2():
      return self.execute_cpu_tf2(compute_fn, inputs)
    else:
      return self.execute_cpu_tf1(compute_fn, inputs)

  def execute_tpu(self, compute_fn, inputs):
    """Executes compute_fn on TPU.

    Depending on the underlying TensorFlow installation (build deps) runs in
    either TF 1.X or TF 2.X style.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single tensor.
    """
    if not self.has_tpu():
      raise ValueError('No TPU Device found.')
    if self.is_tf2():
      return self.execute_tpu_tf2(compute_fn, inputs)
    else:
      return self.execute_tpu_tf1(compute_fn, inputs)

  def execute_tf2(self, compute_fn, inputs):
    """Runs compute_fn with TensorFlow 2.0.

    Executes on TPU if available, otherwise executes on CPU.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single tensor.
    """
    if not self.is_tf2():
      raise ValueError('Required version TensorFlow 2.0 is not available.')
    if self.has_tpu():
      return self.execute_tpu_tf2(compute_fn, inputs)
    else:
      return self.execute_cpu_tf2(compute_fn, inputs)

  def execute_tf1(self, compute_fn, inputs):
    """Runs compute_fn with TensorFlow 1.X.

    Executes on TPU if available, otherwise executes on CPU.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single tensor.
    """
    if self.is_tf2():
      raise ValueError('Required version Tenforflow 1.X is not available.')
    if self.has_tpu():
      return self.execute_tpu_tf1(compute_fn, inputs)
    else:
      return self.execute_cpu_tf1(compute_fn, inputs)

  def execute(self, compute_fn, inputs):
    """Runs compute_fn with inputs and returns results.

    * Executes in either TF1.X or TF2.X style based on the TensorFlow version.
    * Executes on TPU if available, otherwise executes on CPU.

    Args:
      compute_fn: a function containing Tensorflow computation that takes a list
        of input numpy tensors, performs computation and returns output numpy
        tensors.
      inputs: a list of numpy arrays to feed input to the `compute_fn`.

    Returns:
      A list of numpy arrays or a single tensor.
    """
    if self.has_tpu() and tf2.enabled():
      return self.execute_tpu_tf2(compute_fn, inputs)
    elif not self.has_tpu() and tf2.enabled():
      return self.execute_cpu_tf2(compute_fn, inputs)
    elif self.has_tpu() and not tf2.enabled():
      return self.execute_tpu_tf1(compute_fn, inputs)
    else:
      return self.execute_cpu_tf1(compute_fn, inputs)
