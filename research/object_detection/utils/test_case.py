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
"""A convenience wrapper around tf.test.TestCase to enable TPU tests."""

import os
import tensorflow as tf
from tensorflow.contrib import tpu

flags = tf.app.flags

flags.DEFINE_bool('tpu_test', False, 'Whether to configure test for TPU.')
FLAGS = flags.FLAGS


# BEGIN GOOGLE-INTERNAL
def hlo_memory_profile(function):
  """Decorator to set environment variables that produce XLA HLO memory profile.

  Args:
    function: A function to run with XLA HLO profiling on.

  Returns:
    A decorated function that dumps the XLA HLO memory profile in test output
    directory.

  Usage:
    @test_case.hlo_memory_profile
    def test_run_my_tf_tpu_op(self):
      ...

    After running the test, access the memory profile proto from output files
    and generate visualization using XLA memory visualizer.
  """
  def wrapper_func(*args, **kwargs):
    outputs_dir = os.environ['TEST_UNDECLARED_OUTPUTS_DIR']
    path_to_function = os.path.join(outputs_dir, 'hlo_memory_profile',
                                    function.__name__)
    os.environ['TF_XLA_FLAGS'] = (
        '--xla_dump_optimized_hlo_proto_to=' + path_to_function)
    return function(*args, **kwargs)
  return wrapper_func
# END GOOGLE-INTERNAL


class TestCase(tf.test.TestCase):
  """Extends tf.test.TestCase to optionally allow running tests on TPU."""

  def execute_tpu(self, graph_fn, inputs):
    """Constructs the graph, executes it on TPU and returns the result.

    Args:
      graph_fn: a callable that constructs the tensorflow graph to test. The
        arguments of this function should correspond to `inputs`.
      inputs: a list of numpy arrays to feed input to the computation graph.

    Returns:
      A list of numpy arrays or a scalar returned from executing the tensorflow
      graph.
    """
    with self.test_session(graph=tf.Graph()) as sess:
      placeholders = [tf.placeholder_with_default(v, v.shape) for v in inputs]
      tpu_computation = tpu.rewrite(graph_fn, placeholders)
      sess.run(tpu.initialize_system())
      sess.run([tf.global_variables_initializer(), tf.tables_initializer(),
                tf.local_variables_initializer()])
      materialized_results = sess.run(tpu_computation,
                                      feed_dict=dict(zip(placeholders, inputs)))
      sess.run(tpu.shutdown_system())
      if (hasattr(materialized_results, '__len__') and
          len(materialized_results) == 1 and
          (isinstance(materialized_results, list) or
           isinstance(materialized_results, tuple))):
        materialized_results = materialized_results[0]
    return materialized_results

  def execute_cpu(self, graph_fn, inputs):
    """Constructs the graph, executes it on CPU and returns the result.

    Args:
      graph_fn: a callable that constructs the tensorflow graph to test. The
        arguments of this function should correspond to `inputs`.
      inputs: a list of numpy arrays to feed input to the computation graph.

    Returns:
      A list of numpy arrays or a scalar returned from executing the tensorflow
      graph.
    """
    with self.test_session(graph=tf.Graph()) as sess:
      placeholders = [tf.placeholder_with_default(v, v.shape) for v in inputs]
      results = graph_fn(*placeholders)
      sess.run([tf.global_variables_initializer(), tf.tables_initializer(),
                tf.local_variables_initializer()])
      materialized_results = sess.run(results, feed_dict=dict(zip(placeholders,
                                                                  inputs)))

      if (hasattr(materialized_results, '__len__') and
          len(materialized_results) == 1 and
          (isinstance(materialized_results, list) or
           isinstance(materialized_results, tuple))):
        materialized_results = materialized_results[0]
    return materialized_results

  def execute(self, graph_fn, inputs):
    """Constructs the graph, creates a test session and returns the results.

    The graph is executed either on TPU or CPU based on the `tpu_test` flag.

    Args:
      graph_fn: a callable that constructs the tensorflow graph to test. The
        arguments of this function should correspond to `inputs`.
      inputs: a list of numpy arrays to feed input to the computation graph.

    Returns:
      A list of numpy arrays or a scalar returned from executing the tensorflow
      graph.
    """
    if FLAGS.tpu_test:
      return self.execute_tpu(graph_fn, inputs)
    else:
      return self.execute_cpu(graph_fn, inputs)
