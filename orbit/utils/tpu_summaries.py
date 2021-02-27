# Copyright 2021 The Orbit Authors. All Rights Reserved.
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

"""Contains utilities for TPU summary optimization."""

import contextlib
import functools

import tensorflow as tf


@contextlib.contextmanager
def _soft_device_placement():
  """Context manager for soft device placement, allowing summaries on CPU."""
  original_setting = tf.config.get_soft_device_placement()
  try:
    tf.config.set_soft_device_placement(True)
    yield
  finally:
    tf.config.set_soft_device_placement(original_setting)


class OptionalSummariesFunction:
  """Wrapper that provides versions of a function with and without summaries.

  This is a utility class for implementing optimized summary recording via a
  two-function approach, specifically important for TPUs. Two `tf.function`
  versions of a given `function` are created: one with soft device placement
  enabled (for use on steps that require summary writing), and one with summary
  writing and soft device placement entirely disabled (for use on all other
  steps). This removes any performance impact of summaries on steps where they
  aren't recorded (b/148418718).

  This class can be used as a base class to implement summary optimizations for
  a function with a specific signature. For example, to implement efficient TPU
  summaries for a standard `train()` method (as in `orbit.AbstractTrainer`):

      class TrainFunctionWithSummaries(orbit.utils.OptionalSummariesFunction):
        '''Implements a two-program approach for summaries on TPU.'''

        def __call__(self, num_steps):
          if tf.summary.should_record_summaries():
            output = self.with_summaries(tf.constant(1))
            num_steps -= 1
          if num_steps >= 1:
            output = self.without_summaries(num_steps)
          return output

  This can be used directly or to implement a decorator:

      def train_function_with_summaries(function=None, **kwargs):
        if function is not None:
          return TrainFunctionWithSummaries(function, **kwargs)
        return functools.partial(TrainFunctionWithSummaries, **kwargs)

  The director can be applied directly to `train()` methods:

      @train_function_with_summaries
      def train(self, num_steps):
        ...

  A similar approach approach can be implemented for functions with different
  signatures.

  Note: The above approach assumes that the frequency of summary writing is
  based on a step interval that is divisible by the number of steps executed
  in each call to the `train()` function. This is enforced by the
  `orbit.Controller`.

  This wrapper properly handles instance methods (see `__get__`).

  Attributes:
    with_summaries: A wrapped version of the underlying function with summaries
      enabled (using whatever the active predicate is for
      `tf.summary.record_if`), and placed inside a "soft device placement"
      context to enable summary recording on TPU.
    without_summaries: A wrapped version of the underlying function with all
      summary recording disabled.
  """

  def __init__(self, function, **tf_function_kwargs):
    """Constructs an instance wrapping the given `function`.

    The given `function` is wrapped twice: Once in a "soft device placement"
    context (allowing summaries to also run on TPU), and once with summary
    recording entirely disabled.

    Both of these versions are compiled via `tf.function` (optionally using any
    supplied `tf.function` settings), and made available as attributes.

    Args:
      function: The underlying function to wrap.
      **tf_function_kwargs: Additional arguments to pass to `tf.function`.
    """

    @tf.function(**tf_function_kwargs)
    @functools.wraps(function)
    def with_summaries(*args, **kwargs):
      with _soft_device_placement():
        return function(*args, **kwargs)

    @tf.function(**tf_function_kwargs)
    @functools.wraps(function)
    def without_summaries(*args, **kwargs):
      with tf.summary.record_if(False):
        return function(*args, **kwargs)

    self.with_summaries = with_summaries
    self.without_summaries = without_summaries

  def __get__(self, instance, owner):
    """Allows this class to be used to wrap methods as well as free functions.

    For `tf.function` to work properly in all cases (e.g., when an
    input_signature is specified), any `tf.function`-converted methods must be
    properly bound to an instance if they are called as an instance method.

    This is done by implementing this `__get__` method of the descriptor
    protocol, and forwarding to the `__get__` method on the underlying
    `tf.function`s.

    Args:
      instance: The instance to bind to.
      owner: The class type of the instance.

    Returns:
      A new bound instance of `TpuDiscretionarySummariesFunctions`.
    """
    new = object.__new__(self.__class__)
    # pytype: disable=attribute-error  # See b/162476201.
    new.with_summaries = self.with_summaries.__get__(instance, owner)
    new.without_summaries = self.without_summaries.__get__(instance, owner)
    # pytype: enable=attribute-error
    return new
