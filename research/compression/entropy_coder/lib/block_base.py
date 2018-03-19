# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Base class for Tensorflow building blocks."""

import collections
import contextlib
import itertools

import tensorflow as tf

_block_stacks = collections.defaultdict(lambda: [])


class BlockBase(object):
  """Base class for transform wrappers of Tensorflow.

  To implement a Tensorflow transform block, inherit this class.

  1. To create a variable, use NewVar() method. Do not overload this method!
     For example, use as follows.
         a_variable = self.NewVar(initial_value)

  2. All Tensorflow-related code must be done inside 'with self._BlockScope().'
     Otherwise, name scoping and block hierarchy will not work. An exception
     is _Apply() method, which is already called inside the context manager
     by __call__() method.

  3. Override and implement _Apply() method. This method is called by
     __call__() method.

  The users would use blocks like the following.
      nn1 = NN(128, bias=Bias(0), act=tf.nn.relu)
      y = nn1(x)

  Some things to consider.

  - Use lazy-initialization if possible. That is, initialize at first Apply()
    rather than at __init__().

  Note: if needed, the variables can be created on a specific parameter
  server by creating blocks in a scope like:
    with g.device(device):
      linear = Linear(...)
  """

  def __init__(self, name):
    self._variables = []
    self._subblocks = []
    self._called = False

    # Intentionally distinguishing empty string and None.
    # If name is an empty string, then do not use name scope.
    self.name = name if name is not None else self.__class__.__name__
    self._graph = tf.get_default_graph()

    if self.name:
      # Capture the scope string at the init time.
      with self._graph.name_scope(self.name) as scope:
        self._scope_str = scope
    else:
      self._scope_str = ''

    # Maintain hierarchy structure of blocks.
    self._stack = _block_stacks[self._graph]
    if self.__class__ is BlockBase:
      # This code is only executed to create the root, which starts in the
      # initialized state.
      assert not self._stack
      self._parent = None
      self._called = True  # The root is initialized.
      return

    # Create a fake root if a root is not already present.
    if not self._stack:
      self._stack.append(BlockBase('NoOpRoot'))

    self._parent = self._stack[-1]
    self._parent._subblocks.append(self)  # pylint: disable=protected-access

  def __repr__(self):
    return '"{}" ({})'.format(self._scope_str, self.__class__.__name__)

  @contextlib.contextmanager
  def _OptionalNameScope(self, scope_str):
    if scope_str:
      with self._graph.name_scope(scope_str):
        yield
    else:
      yield

  @contextlib.contextmanager
  def _BlockScope(self):
    """Context manager that handles graph, namescope, and nested blocks."""
    self._stack.append(self)

    try:
      with self._graph.as_default():
        with self._OptionalNameScope(self._scope_str):
          yield self
    finally:  # Pop from the stack no matter exception is raised or not.
      # The following line is executed when leaving 'with self._BlockScope()'
      self._stack.pop()

  def __call__(self, *args, **kwargs):
    assert self._stack is _block_stacks[self._graph]

    with self._BlockScope():
      ret = self._Apply(*args, **kwargs)

    self._called = True
    return ret

  def _Apply(self, *args, **kwargs):
    """Implementation of __call__()."""
    raise NotImplementedError()

  # Redirect all variable creation to this single function, so that we can
  # switch to better variable creation scheme.
  def NewVar(self, value, **kwargs):
    """Creates a new variable.

    This function creates a variable, then returns a local copy created by
    Identity operation. To get the Variable class object, use LookupRef()
    method.

    Note that each time Variable class object is used as an input to an
    operation, Tensorflow will create a new Send/Recv pair. This hurts
    performance.

    If not for assign operations, use the local copy returned by this method.

    Args:
      value: Initialization value of the variable. The shape and the data type
        of the variable is determined by this initial value.
      **kwargs: Extra named arguments passed to Variable.__init__().

    Returns:
      A local copy of the new variable.
    """
    v = tf.Variable(value, **kwargs)

    self._variables.append(v)
    return v

  @property
  def initialized(self):
    """Returns bool if the block is initialized.

    By default, BlockBase assumes that a block is initialized when __call__()
    is executed for the first time. If this is an incorrect assumption for some
    subclasses, override this property in those subclasses.

    Returns:
      True if initialized, False otherwise.
    """
    return self._called

  def AssertInitialized(self):
    """Asserts initialized property."""
    if not self.initialized:
      raise RuntimeError('{} has not been initialized.'.format(self))

  def VariableList(self):
    """Returns the list of all tensorflow variables used inside this block."""
    variables = list(itertools.chain(
        itertools.chain.from_iterable(
            t.VariableList() for t in self._subblocks),
        self._VariableList()))
    return variables

  def _VariableList(self):
    """Returns the list of all tensorflow variables owned by this block."""
    self.AssertInitialized()
    return self._variables

  def CreateWeightLoss(self):
    """Returns L2 loss list of (almost) all variables used inside this block.

    When this method needs to be overridden, there are two choices.

    1. Override CreateWeightLoss() to change the weight loss of all variables
       that belong to this block, both directly and indirectly.
    2. Override _CreateWeightLoss() to change the weight loss of all
       variables that directly belong to this block but not to the sub-blocks.

    Returns:
      A Tensor object or None.
    """
    losses = list(itertools.chain(
        itertools.chain.from_iterable(
            t.CreateWeightLoss() for t in self._subblocks),
        self._CreateWeightLoss()))
    return losses

  def _CreateWeightLoss(self):
    """Returns weight loss list of variables that belong to this block."""
    self.AssertInitialized()
    with self._BlockScope():
      return [tf.nn.l2_loss(v) for v in self._variables]

  def CreateUpdateOps(self):
    """Creates update operations for this block and its sub-blocks."""
    ops = list(itertools.chain(
        itertools.chain.from_iterable(
            t.CreateUpdateOps() for t in self._subblocks),
        self._CreateUpdateOps()))
    return ops

  def _CreateUpdateOps(self):
    """Creates update operations for this block."""
    self.AssertInitialized()
    return []

  def MarkAsNonTrainable(self):
    """Mark all the variables of this block as non-trainable.

    All the variables owned directly or indirectly (through subblocks) are
    marked as non trainable.

    This function along with CheckpointInitOp can be used to load a pretrained
    model that consists in only one part of the whole graph.
    """
    assert self._called

    all_variables = self.VariableList()
    collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in all_variables:
      if v in collection:
        collection.remove(v)


def CreateWeightLoss():
  """Returns all weight losses from the blocks in the graph."""
  stack = _block_stacks[tf.get_default_graph()]
  if not stack:
    return []
  return stack[0].CreateWeightLoss()


def CreateBlockUpdates():
  """Combines all updates from the blocks in the graph."""
  stack = _block_stacks[tf.get_default_graph()]
  if not stack:
    return []
  return stack[0].CreateUpdateOps()
