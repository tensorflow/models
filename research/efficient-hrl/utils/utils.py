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

"""TensorFlow utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import tensorflow as tf
from tf_agents import specs
from tf_agents.utils import common

_tf_print_counts = dict()
_tf_print_running_sums = dict()
_tf_print_running_counts = dict()
_tf_print_ids = 0


def get_contextual_env_base(env_base, begin_ops=None, end_ops=None):
  """Wrap env_base with additional tf ops."""
  # pylint: disable=protected-access
  def init(self_, env_base):
    self_._env_base = env_base
    attribute_list = ["_render_mode", "_gym_env"]
    for attribute in attribute_list:
      if hasattr(env_base, attribute):
        setattr(self_, attribute, getattr(env_base, attribute))
    if hasattr(env_base, "physics"):
      self_._physics = env_base.physics
    elif hasattr(env_base, "gym"):
      class Physics(object):
        def render(self, *args, **kwargs):
          return env_base.gym.render("rgb_array")
      physics = Physics()
      self_._physics = physics
      self_.physics = physics
  def set_sess(self_, sess):
    self_._sess = sess
    if hasattr(self_._env_base, "set_sess"):
      self_._env_base.set_sess(sess)
  def begin_episode(self_):
    self_._env_base.reset()
    if begin_ops is not None:
      self_._sess.run(begin_ops)
  def end_episode(self_):
    self_._env_base.reset()
    if end_ops is not None:
      self_._sess.run(end_ops)
  return type("ContextualEnvBase", (env_base.__class__,), dict(
      __init__=init,
      set_sess=set_sess,
      begin_episode=begin_episode,
      end_episode=end_episode,
  ))(env_base)
  # pylint: enable=protected-access


def merge_specs(specs_):
  """Merge TensorSpecs.

  Args:
    specs_: List of TensorSpecs to be merged.
  Returns:
    a TensorSpec: a merged TensorSpec.
  """
  shape = specs_[0].shape
  dtype = specs_[0].dtype
  name = specs_[0].name
  for spec in specs_[1:]:
    assert shape[1:] == spec.shape[1:], "incompatible shapes: %s, %s" % (
        shape, spec.shape)
    assert dtype == spec.dtype, "incompatible dtypes: %s, %s" % (
        dtype, spec.dtype)
    shape = merge_shapes((shape, spec.shape), axis=0)
  return specs.TensorSpec(
      shape=shape,
      dtype=dtype,
      name=name,
  )


def merge_shapes(shapes, axis=0):
  """Merge TensorShapes.

  Args:
    shapes: List of TensorShapes to be merged.
    axis: optional, the axis to merge shaped.
  Returns:
    a TensorShape: a merged TensorShape.
  """
  assert len(shapes) > 1
  dims = deepcopy(shapes[0].dims)
  for shape in shapes[1:]:
    assert shapes[0].ndims == shape.ndims
    dims[axis] += shape.dims[axis]
  return tf.TensorShape(dims=dims)


def get_all_vars(ignore_scopes=None):
  """Get all tf variables in scope.

  Args:
    ignore_scopes: A list of scope names to ignore.
  Returns:
    A list of all tf variables in scope.
  """
  all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  all_vars = [var for var in all_vars if ignore_scopes is None or not
              any(var.name.startswith(scope) for scope in ignore_scopes)]
  return all_vars


def clip(tensor, range_=None):
  """Return a tf op which clips tensor according to range_.

  Args:
    tensor: A Tensor to be clipped.
    range_: None, or a tuple representing (minval, maxval)
  Returns:
    A clipped Tensor.
  """
  if range_ is None:
    return tf.identity(tensor)
  elif isinstance(range_, (tuple, list)):
    assert len(range_) == 2
    return tf.clip_by_value(tensor, range_[0], range_[1])
  else: raise NotImplementedError("Unacceptable range input: %r" % range_)


def clip_to_bounds(value, minimum, maximum):
  """Clips value to be between minimum and maximum.

  Args:
    value: (tensor) value to be clipped.
    minimum: (numpy float array) minimum value to clip to.
    maximum: (numpy float array) maximum value to clip to.
  Returns:
    clipped_value: (tensor) `value` clipped to between `minimum` and `maximum`.
  """
  value = tf.minimum(value, maximum)
  return tf.maximum(value, minimum)


clip_to_spec = common.clip_to_spec
def _clip_to_spec(value, spec):
  """Clips value to a given bounded tensor spec.

  Args:
    value: (tensor) value to be clipped.
    spec: (BoundedTensorSpec) spec containing min. and max. values for clipping.
  Returns:
    clipped_value: (tensor) `value` clipped to be compatible with `spec`.
  """
  return clip_to_bounds(value, spec.minimum, spec.maximum)


join_scope = common.join_scope
def _join_scope(parent_scope, child_scope):
  """Joins a parent and child scope using `/`, checking for empty/none.

  Args:
    parent_scope: (string) parent/prefix scope.
    child_scope: (string) child/suffix scope.
  Returns:
    joined scope: (string) parent and child scopes joined by /.
  """
  if not parent_scope:
    return child_scope
  if not child_scope:
    return parent_scope
  return '/'.join([parent_scope, child_scope])


def assign_vars(vars_, values):
  """Returns the update ops for assigning a list of vars.

  Args:
    vars_: A list of variables.
    values: A list of tensors representing new values.
  Returns:
    A list of update ops for the variables.
  """
  return [var.assign(value) for var, value in zip(vars_, values)]


def identity_vars(vars_):
  """Return the identity ops for a list of tensors.

  Args:
    vars_: A list of tensors.
  Returns:
    A list of identity ops.
  """
  return [tf.identity(var) for var in vars_]


def tile(var, batch_size=1):
  """Return tiled tensor.

  Args:
    var: A tensor representing the state.
    batch_size: Batch size.
  Returns:
    A tensor with shape [batch_size,] + var.shape.
  """
  batch_var = tf.tile(
      tf.expand_dims(var, 0),
      (batch_size,) + (1,) * var.get_shape().ndims)
  return batch_var


def batch_list(vars_list):
  """Batch a list of variables.

  Args:
    vars_list: A list of tensor variables.
  Returns:
    A list of tensor variables with additional first dimension.
  """
  return [tf.expand_dims(var, 0) for var in vars_list]


def tf_print(op,
             tensors,
             message="",
             first_n=-1,
             name=None,
             sub_messages=None,
             print_freq=-1,
             include_count=True):
  """tf.Print, but to stdout."""
  # TODO(shanegu): `name` is deprecated. Remove from the rest of codes.
  global _tf_print_ids
  _tf_print_ids += 1
  name = _tf_print_ids
  _tf_print_counts[name] = 0
  if print_freq > 0:
    _tf_print_running_sums[name] = [0 for _ in tensors]
    _tf_print_running_counts[name] = 0
  def print_message(*xs):
    """print message fn."""
    _tf_print_counts[name] += 1
    if print_freq > 0:
      for i, x in enumerate(xs):
        _tf_print_running_sums[name][i] += x
      _tf_print_running_counts[name] += 1
    if (print_freq <= 0 or _tf_print_running_counts[name] >= print_freq) and (
        first_n < 0 or _tf_print_counts[name] <= first_n):
      for i, x in enumerate(xs):
        if print_freq > 0:
          del x
          x = _tf_print_running_sums[name][i]/_tf_print_running_counts[name]
        if sub_messages is None:
          sub_message = str(i)
        else:
          sub_message = sub_messages[i]
        log_message = "%s, %s" % (message, sub_message)
        if include_count:
          log_message += ", count=%d" % _tf_print_counts[name]
        tf.logging.info("[%s]: %s" % (log_message, x))
      if print_freq > 0:
        for i, x in enumerate(xs):
          _tf_print_running_sums[name][i] = 0
        _tf_print_running_counts[name] = 0
    return xs[0]

  print_op = tf.py_func(print_message, tensors, tensors[0].dtype)
  with tf.control_dependencies([print_op]):
    op = tf.identity(op)
  return op


periodically = common.periodically
def _periodically(body, period, name='periodically'):
  """Periodically performs a tensorflow op."""
  if period is None or period == 0:
    return tf.no_op()

  if period < 0:
    raise ValueError("period cannot be less than 0.")

  if period == 1:
    return body()

  with tf.variable_scope(None, default_name=name):
    counter = tf.get_variable(
        "counter",
        shape=[],
        dtype=tf.int64,
        trainable=False,
        initializer=tf.constant_initializer(period, dtype=tf.int64))

    def _wrapped_body():
      with tf.control_dependencies([body()]):
        return counter.assign(1)

    update = tf.cond(
        tf.equal(counter, period), _wrapped_body,
        lambda: counter.assign_add(1))

  return update

soft_variables_update = common.soft_variables_update
