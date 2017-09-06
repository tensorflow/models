# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""Implementation of the ModelAdapter class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow as tf

from learned_optimizer.problems import problem_generator as pg


class ModelAdapter(pg.Problem):
  """Adapts Tensorflow models/graphs into a form suitable for meta-training.

  This class adapts an existing TensorFlow graph into a form suitable for
  meta-training a learned optimizer.
  """

  def __init__(self, make_loss_and_init_fn):
    """Wraps a model in the Problem interface.

    make_loss_and_init argument is a callable that returns a tuple of
    two other callables as follows.

    The first will construct most of the graph and return the problem loss. It
    is essential that this graph contains the totality of the model's variables,
    but none of its queues.

    The second will return construct the model initialization graph given a list
    of parameters and return a callable that is passed an instance of
    tf.Session, and should initialize the models' parameters.

    An argument value function would look like this:

    ```python
    def make_loss_and_init_fn():
      inputs = queued_reader()

      def make_loss():
        return create_model_with_variables(inputs)

      def make_init_fn(parameters):
        saver = tf.Saver(parameters)
        def init_fn(sess):
          sess.restore(sess, ...)
        return init_fn

      return make_loss, make_init_fn
    ```

    Args:
      make_loss_and_init_fn: a callable, as described aboce
    """
    make_loss_fn, make_init_fn = make_loss_and_init_fn()

    self.make_loss_fn = make_loss_fn
    self.parameters, self.constants = _get_variables(make_loss_fn)

    if make_init_fn is not None:
      init_fn = make_init_fn(self.parameters + self.constants)
    else:
      init_op = tf.initialize_variables(self.parameters + self.constants)
      init_fn = lambda sess: sess.run(init_op)

    tf.logging.info("ModelAdapter parameters: %s",
                    [op.name for op in self.parameters])
    tf.logging.info("ModelAdapter constants: %s",
                    [op.name for op in self.constants])

    super(ModelAdapter, self).__init__(
        [], random_seed=None, noise_stdev=0.0, init_fn=init_fn)

  def init_tensors(self, seed=None):
    """Returns a list of tensors with the given shape."""
    return self.parameters

  def init_variables(self, seed=None):
    """Returns a list of variables with the given shape."""
    # NOTE(siege): This is awkward, as these are not set as trainable.
    return self.parameters

  def objective(self, parameters, data=None, labels=None):
    """Computes the objective given a list of parameters.

    Args:
      parameters: The parameters to optimize (as a list of tensors)
      data: An optional batch of data for calculating objectives
      labels: An optional batch of corresponding labels

    Returns:
      A scalar tensor representing the objective value
    """
    # We need to set up a mapping based on the original parameter names, because
    # the parameters passed can be arbitrary tensors.
    parameter_mapping = {
        old_p.name: p
        for old_p, p in zip(self.parameters, parameters)
    }

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      return _make_with_custom_variables(self.make_loss_fn, parameter_mapping)


def _get_variables(func):
  """Calls func, returning any variables created.

  The created variables are modified to not be trainable, and are placed into
  the LOCAL_VARIABLES collection.

  Args:
    func: Function to be called.

  Returns:
    A tuple (variables, constants) where the first element is a list of
    trainable variables and the second is the non-trainable variables.
  """
  variables = []
  constants = []

  # We need to create these variables like normal, so grab the original
  # constructor before we mock it.
  original_init = tf.Variable.__init__

  def custom_init(self, *args, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    # Making these variables local keeps them out of the optimizer's checkpoints
    # somehow.
    kwargs["collections"] = [tf.GraphKeys.LOCAL_VARIABLES]
    original_init(self, *args, **kwargs)
    if trainable:
      variables.append(self)
    else:
      constants.append(self)

  # This name-scope is just a nicety for TensorBoard.
  with tf.name_scope("unused_graph"):
    with mock.patch.object(tf.Variable, "__init__", custom_init):
      func()

  return variables, constants


def _make_with_custom_variables(func, variable_mapping):
  """Calls func and replaces the value of some variables created in it.

  Args:
    func: Function to be called.
    variable_mapping: A mapping of variable name to the replacement tensor or
      tf.Variable.

  Returns:
    The return value of func is returned.
  """
  original_value = tf.Variable.value

  def custom_value(self):
    if self.name in variable_mapping:
      replacement = variable_mapping[self.name]
      tf.logging.info("Replaced %s with %s" % (self.name, replacement))

      # value() method needs to return a tensor, we need to call value on it.
      # This has to be done manually like this otherwise we'll get an infinite
      # loop.
      if isinstance(replacement, tf.Variable):
        replacement = original_value(replacement)

      return replacement
    else:
      return original_value(self)

  with mock.patch.object(tf.Variable, "value", custom_value):
    with mock.patch.object(tf.Variable, "_AsTensor", custom_value):
      return func()
