# Copyright 2018 Google, Inc. All Rights Reserved.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
import numpy as np
import collections
from learning_unsupervised_learning import utils

from tensorflow.python.util import nest

from learning_unsupervised_learning import variable_replace


class LinearBatchNorm(snt.AbstractModule):
  """Module that does a Linear layer then a BatchNorm followed by an activation fn"""
  def __init__(self, size, activation_fn=tf.nn.relu, name="LinearBatchNorm"):
    self.size = size
    self.activation_fn = activation_fn
    super(LinearBatchNorm, self).__init__(name=name)

  def _build(self, x):
    x = tf.to_float(x)
    initializers={"w": tf.truncated_normal_initializer(stddev=0.01)}
    lin = snt.Linear(self.size, use_bias=False, initializers=initializers)
    z = lin(x)

    scale = tf.constant(1., dtype=tf.float32)
    offset = tf.get_variable(
        "b",
        shape=[1, z.shape.as_list()[1]],
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        dtype=tf.float32
    )

    mean, var = tf.nn.moments(z, [0], keep_dims=True)
    z = ((z - mean) * tf.rsqrt(var + 1e-6)) * scale + offset

    x_p = self.activation_fn(z)

    return z, x_p

  # This needs to work by string name sadly due to how the variable replace
  # works and would also work even if the custom getter approuch was used.
  # This is verbose, but it should atleast be clear as to what is going on.
  # TODO(lmetz) a better way to do this (the next 3 functions:
  #    _raw_name, w(), b() )
  def _raw_name(self, var_name):
    """Return just the name of the variable, not the scopes."""
    return var_name.split("/")[-1].split(":")[0]


  @property
  def w(self):
    var_list = snt.get_variables_in_module(self)
    w = [x for x in var_list if self._raw_name(x.name) == "w"]
    assert len(w) == 1
    return w[0]

  @property
  def b(self):
    var_list = snt.get_variables_in_module(self)
    b = [x for x in var_list if self._raw_name(x.name) == "b"]
    assert len(b) == 1
    return b[0]



class Linear(snt.AbstractModule):
  def __init__(self, size, use_bias=True, init_const_mag=True):
    self.size = size
    self.use_bias = use_bias
    self.init_const_mag = init_const_mag
    super(Linear, self).__init__(name="commonLinear")

  def _build(self, x):
    if self.init_const_mag:
      initializers={"w": tf.truncated_normal_initializer(stddev=0.01)}
    else:
      initializers={}
    lin = snt.Linear(self.size, use_bias=self.use_bias, initializers=initializers)
    z = lin(x)
    return z

  # This needs to work by string name sadly due to how the variable replace
  # works and would also work even if the custom getter approuch was used.
  # This is verbose, but it should atleast be clear as to what is going on.
  # TODO(lmetz) a better way to do this (the next 3 functions:
  #    _raw_name, w(), b() )
  def _raw_name(self, var_name):
    """Return just the name of the variable, not the scopes."""
    return var_name.split("/")[-1].split(":")[0]

  @property
  def w(self):
    var_list = snt.get_variables_in_module(self)
    if self.use_bias:
      assert len(var_list) == 2, "Found not 2 but %d" % len(var_list)
    else:
      assert len(var_list) == 1, "Found not 1 but %d" % len(var_list)
    w = [x for x in var_list if self._raw_name(x.name) == "w"]
    assert len(w) == 1
    return w[0]

  @property
  def b(self):
    var_list = snt.get_variables_in_module(self)
    assert len(var_list) == 2, "Found not 2 but %d" % len(var_list)
    b = [x for x in var_list if self._raw_name(x.name) == "b"]
    assert len(b) == 1
    return b[0]


def transformer_at_state(base_model, new_variables):
  """Get the base_model that has been transformed to use the variables
  in final_state.
  Args:
    base_model: snt.Module
      Goes from batch to features
    new_variables: list
      New list of variables to use
  Returns:
    func: callable of same api as base_model.
  """
  assert not variable_replace.in_variable_replace_scope()

  def _feature_transformer(input_data):
    """Feature transformer at the end of training."""
    initial_variables = base_model.get_variables()
    replacement = collections.OrderedDict(
        utils.eqzip(initial_variables, new_variables))
    with variable_replace.variable_replace(replacement):
      features = base_model(input_data)
    return features

  return _feature_transformer
