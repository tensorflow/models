# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow testing subclass to automate numerical testing.

Reference tests determine when behavior deviates from some "gold standard," and
are useful for determining when layer definitions have changed without
performing full regression testing, which is generally prohibitive. This class
handles the symbolic graph comparison as well as loading weights to avoid
relying on random number generation, which can change.

The tests performed by this class are:

1) Compare a generated graph against a reference graph. Differences are not
   necessarily fatal.
2) Attempt to load known weights for the graph. If this step succeeds but
   changes are present in the graph, a warning is issued but does not raise
   an exception.
3) Perform a calculation and compare the result to a reference value.

This class also provides a method to generate reference data.

Note:
  The test class is responsible for fixing the random seed during graph
  definition. A convenience method name_to_seed() is provided to make this
  process easier.

The test class should also define a .regenerate() class method which (usually)
just calls the op definition function with test=False for all relevant tests.

A concise example of this class in action is provided in:
  official/utils/testing/reference_data_test.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class BaseTest(tf.test.TestCase):
  """TestCase subclass for performing reference data tests."""

  def regenerate(self):
    """Subclasses should override this function to generate a new reference."""
    raise NotImplementedError

  @property
  def test_name(self):
    """Subclass should define its own name."""
    raise NotImplementedError

  @property
  def data_root(self):
    """Use the subclass directory rather than the parent directory.

    Returns:
      The path prefix for reference data.
    """
    return os.path.join(os.path.split(
        os.path.abspath(__file__))[0], "reference_data", self.test_name)

  ckpt_prefix = "model.ckpt"

  @staticmethod
  def name_to_seed(name):
    """Convert a string into a 32 bit integer.

    This function allows test cases to easily generate random fixed seeds by
    hashing the name of the test. The hash string is in hex rather than base 10
    which is why there is a 16 in the int call, and the modulo projects the
    seed from a 128 bit int to 32 bits for readability.

    Args:
      name: A string containing the name of a test.

    Returns:
      A pseudo-random 32 bit integer derived from name.
    """
    seed = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(seed, 16) % (2**32 - 1)

  @staticmethod
  def common_tensor_properties(input_array):
    """Convenience function for matrix testing.

    In tests we wish to determine whether a result has changed. However storing
    an entire n-dimensional array is impractical. A better approach is to
    calculate several values from that array and test that those derived values
    are unchanged. The properties themselves are arbitrary and should be chosen
    to be good proxies for a full equality test.

    Args:
      input_array: A numpy array from which key values are extracted.

    Returns:
      A list of values derived from the input_array for equality tests.
    """
    output = list(input_array.shape)
    flat_array = input_array.flatten()
    output.extend([float(i) for i in
                   [flat_array[0], flat_array[-1], np.sum(flat_array)]])
    return output

  def default_correctness_function(self, *args):
    """Returns a vector with the concatenation of common properties.

    This function simply calls common_tensor_properties() for every element.
    It is useful as it allows one to easily construct tests of layers without
    having to worry about the details of result checking.

    Args:
      *args: A list of numpy arrays corresponding to tensors which have been
        evaluated.

    Returns:
      A list of values containing properties for every element in args.
    """
    output = []
    for arg in args:
      output.extend(self.common_tensor_properties(arg))
    return output

  def _construct_and_save_reference_files(
      self, name, graph, ops_to_eval, correctness_function):
    """Save reference data files.

    Constructs a serialized graph_def, layer weights, and computation results.
    It then saves them to files which are read at test time.

    Args:
      name: String defining the run. This will be used to define folder names
        and will be used for random seed construction.
      graph: The graph in which the test is conducted.
      ops_to_eval: Ops which the user wishes to be evaluated under a controlled
        session.
      correctness_function: This function accepts the evaluated results of
        ops_to_eval, and returns a list of values. This list must be JSON
        serializable; in particular it is up to the user to convert numpy
        dtypes into builtin dtypes.
    """
    data_dir = os.path.join(self.data_root, name)

    # Make sure there is a clean space for results.
    if os.path.exists(data_dir):
      shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    # Serialize graph for comparison.
    graph_bytes = graph.as_graph_def().SerializeToString()
    expected_file = os.path.join(data_dir, "expected_graph")
    with tf.gfile.Open(expected_file, "wb") as f:
      f.write(graph_bytes)

    with graph.as_default():
      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

    with self.test_session(graph=graph) as sess:
      sess.run(init)
      saver.save(sess=sess, save_path=os.path.join(data_dir, self.ckpt_prefix))

      # These files are not needed for this test.
      os.remove(os.path.join(data_dir, "checkpoint"))
      os.remove(os.path.join(data_dir, self.ckpt_prefix + ".meta"))

      # ops are evaluated even if there is no correctness function to ensure
      # that they can be evaluated.
      eval_results = [op.eval() for op in ops_to_eval]

      if correctness_function is not None:
        results = correctness_function(*eval_results)
        with tf.gfile.Open(os.path.join(data_dir, "results.json"), "w") as f:
          json.dump(results, f)

      with tf.gfile.Open(os.path.join(data_dir, "tf_version.json"), "w") as f:
        json.dump([tf.VERSION, tf.GIT_VERSION], f)

  def _evaluate_test_case(self, name, graph, ops_to_eval, correctness_function):
    """Determine if a graph agrees with the reference data.

    Args:
      name: String defining the run. This will be used to define folder names
        and will be used for random seed construction.
      graph: The graph in which the test is conducted.
      ops_to_eval: Ops which the user wishes to be evaluated under a controlled
        session.
      correctness_function: This function accepts the evaluated results of
        ops_to_eval, and returns a list of values. This list must be JSON
        serializable; in particular it is up to the user to convert numpy
        dtypes into builtin dtypes.
    """
    data_dir = os.path.join(self.data_root, name)

    # Serialize graph for comparison.
    graph_bytes = graph.as_graph_def().SerializeToString()
    expected_file = os.path.join(data_dir, "expected_graph")
    with tf.gfile.Open(expected_file, "rb") as f:
      expected_graph_bytes = f.read()
      # The serialization is non-deterministic byte-for-byte. Instead there is
      # a utility which evaluates the semantics of the two graphs to test for
      # equality. This has the added benefit of providing some information on
      # what changed.
      #   Note: The summary only show the first difference detected. It is not
      #         an exhaustive summary of differences.
    differences = pywrap_tensorflow.EqualGraphDefWrapper(
        graph_bytes, expected_graph_bytes).decode("utf-8")

    with graph.as_default():
      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

    with tf.gfile.Open(os.path.join(data_dir, "tf_version.json"), "r") as f:
      tf_version_reference, tf_git_version_reference = json.load(f)  # pylint: disable=unpacking-non-sequence

    tf_version_comparison = ""
    if tf.GIT_VERSION != tf_git_version_reference:
      tf_version_comparison = (
          "Test was built using:     {} (git = {})\n"
          "Local TensorFlow version: {} (git = {})"
          .format(tf_version_reference, tf_git_version_reference,
                  tf.VERSION, tf.GIT_VERSION)
      )

    with self.test_session(graph=graph) as sess:
      sess.run(init)
      try:
        saver.restore(sess=sess, save_path=os.path.join(
            data_dir, self.ckpt_prefix))
        if differences:
          tf.logging.warn(
              "The provided graph is different than expected:\n  {}\n"
              "However the weights were still able to be loaded.\n{}".format(
                  differences, tf_version_comparison)
          )
      except:  # pylint: disable=bare-except
        raise self.failureException(
            "Weight load failed. Graph comparison:\n  {}{}"
            .format(differences, tf_version_comparison))

      eval_results = [op.eval() for op in ops_to_eval]
      if correctness_function is not None:
        results = correctness_function(*eval_results)
        with tf.gfile.Open(os.path.join(data_dir, "results.json"), "r") as f:
          expected_results = json.load(f)
        self.assertAllClose(results, expected_results)

  def _save_or_test_ops(self, name, graph, ops_to_eval=None, test=True,
                        correctness_function=None):
    """Utility function to automate repeated work of graph checking and saving.

    The philosophy of this function is that the user need only define ops on
    a graph and specify which results should be validated. The actual work of
    managing snapshots and calculating results should be automated away.

    Args:
      name: String defining the run. This will be used to define folder names
        and will be used for random seed construction.
      graph: The graph in which the test is conducted.
      ops_to_eval: Ops which the user wishes to be evaluated under a controlled
        session.
      test: Boolean. If True this function will test graph correctness, load
        weights, and compute numerical values. If False the necessary test data
        will be generated and saved.
      correctness_function: This function accepts the evaluated results of
        ops_to_eval, and returns a list of values. This list must be JSON
        serializable; in particular it is up to the user to convert numpy
        dtypes into builtin dtypes.
    """

    ops_to_eval = ops_to_eval or []

    if test:
      try:
        self._evaluate_test_case(
            name=name, graph=graph, ops_to_eval=ops_to_eval,
            correctness_function=correctness_function
        )
      except:
        tf.logging.error("Failed unittest {}".format(name))
        raise
    else:
      self._construct_and_save_reference_files(
          name=name, graph=graph, ops_to_eval=ops_to_eval,
          correctness_function=correctness_function
      )


class ReferenceDataActionParser(argparse.ArgumentParser):
  """Minimal arg parser so that test regeneration can be called from the CLI."""

  def __init__(self):
    super(ReferenceDataActionParser, self).__init__()
    self.add_argument(
        "--regenerate", "-regen",
        action="store_true",
        help="Enable this flag to regenerate test data. If not set unit tests"
             "will be run."
    )


def main(argv, test_class):
  """Simple switch function to allow test regeneration from the CLI."""
  flags = ReferenceDataActionParser().parse_args(argv[1:])
  if flags.regenerate:
    if sys.version_info[0] == 2:
      raise NameError("\nPython2 unittest does not support being run as a "
                      "standalone class.\nAs a result tests must be "
                      "regenerated using Python3.\n"
                      "Tests can be run under 2 or 3.")
    test_class().regenerate()
  else:
    tf.test.main()
