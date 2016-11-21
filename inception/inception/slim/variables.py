# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for creating variables in TF-Slim.

The variables module is typically used for defining model variables from the
ops routines (see slim.ops). Such variables are used for training, evaluation
and inference of models.

All the variables created through this module would be added to the
MODEL_VARIABLES collection, if you create a model variable outside slim, it can
be added with slim.variables.add_variable(external_variable, reuse).

Usage:
  weights_initializer = tf.truncated_normal_initializer(stddev=0.01)
  l2_regularizer = lambda t: losses.l2_loss(t, weight=0.0005)
  weights = variables.variable('weights',
                               shape=[100, 100],
                               initializer=weights_initializer,
                               regularizer=l2_regularizer,
                               device='/cpu:0')

  biases = variables.variable('biases',
                              shape=[100],
                              initializer=tf.zeros_initializer,
                              device='/cpu:0')

  # More complex example.

  net = slim.ops.conv2d(input, 32, [3, 3], scope='conv1')
  net = slim.ops.conv2d(net, 64, [3, 3], scope='conv2')
  with slim.arg_scope([variables.variable], restore=False):
    net = slim.ops.conv2d(net, 64, [3, 3], scope='conv3')

  # Get all model variables from all the layers.
  model_variables = slim.variables.get_variables()

  # Get all model variables from a specific the layer, i.e 'conv1'.
  conv1_variables = slim.variables.get_variables('conv1')

  # Get all weights from all the layers.
  weights = slim.variables.get_variables_by_name('weights')

  # Get all bias from all the layers.
  biases = slim.variables.get_variables_by_name('biases')

  # Get all variables to restore.
  # (i.e. only those created by 'conv1' and 'conv2')
  variables_to_restore = slim.variables.get_variables_to_restore()

************************************************
* Initializing model variables from a checkpoint
************************************************

# Create some variables.
v1 = slim.variables.variable(name="v1", ..., restore=False)
v2 = slim.variables.variable(name="v2", ...) # By default restore=True
...
# The list of variables to restore should only contain 'v2'.
variables_to_restore = slim.variables.get_variables_to_restore()
restorer = tf.train.Saver(variables_to_restore)
with tf.Session() as sess:
  # Restore variables from disk.
  restorer.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import scopes

# Collection containing all the variables created using slim.variables
MODEL_VARIABLES = '_model_variables_'

# Collection containing the slim.variables that are created with restore=True.
VARIABLES_TO_RESTORE = '_variables_to_restore_'


def add_variable(var, restore=True):
  """Adds a variable to the MODEL_VARIABLES collection.

    Optionally it will add the variable to  the VARIABLES_TO_RESTORE collection.
  Args:
    var: a variable.
    restore: whether the variable should be added to the
      VARIABLES_TO_RESTORE collection.

  """
  collections = [MODEL_VARIABLES]
  if restore:
    collections.append(VARIABLES_TO_RESTORE)
  for collection in collections:
    if var not in tf.get_collection(collection):
      tf.add_to_collection(collection, var)


def get_variables(scope=None, suffix=None):
  """Gets the list of variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a copied list of variables with scope and suffix.
  """
  candidates = tf.get_collection(MODEL_VARIABLES, scope)[:]
  if suffix is not None:
    candidates = [var for var in candidates if var.op.name.endswith(suffix)]
  return candidates


def get_variables_to_restore():
  """Gets the list of variables to restore.

  Returns:
    a copied list of variables.
  """
  return tf.get_collection(VARIABLES_TO_RESTORE)[:]


def get_variables_by_name(given_name, scope=None):
  """Gets the list of variables that were given that name.

  Args:
    given_name: name given to the variable without scope.
    scope: an optional scope for filtering the variables to return.

  Returns:
    a copied list of variables with the given name and prefix.
  """
  return get_variables(scope=scope, suffix=given_name)


def get_unique_variable(name):
  """Gets the variable uniquely identified by that name.

  Args:
    name: a name that uniquely identifies the variable.

  Returns:
    a tensorflow variable.

  Raises:
    ValueError: if no variable uniquely identified by the name exists.
  """
  candidates = tf.get_collection(tf.GraphKeys.VARIABLES, name)
  if not candidates:
    raise ValueError('Couldnt find variable %s' % name)

  for candidate in candidates:
    if candidate.op.name == name:
      return candidate
  raise ValueError('Variable %s does not uniquely identify a variable', name)


class VariableDeviceChooser(object):
  """Slim device chooser for variables.

  When using a parameter server it will assign them in a round-robin fashion.
  When not using a parameter server it allows GPU:0 placement otherwise CPU:0.
  """

  def __init__(self,
               num_parameter_servers=0,
               ps_device='/job:ps',
               placement='CPU:0'):
    """Initialize VariableDeviceChooser.

    Args:
      num_parameter_servers: number of parameter servers.
      ps_device: string representing the parameter server device.
      placement: string representing the placement of the variable either CPU:0
        or GPU:0. When using parameter servers forced to CPU:0.
    """
    self._num_ps = num_parameter_servers
    self._ps_device = ps_device
    self._placement = placement if num_parameter_servers == 0 else 'CPU:0'
    self._next_task_id = 0

  def __call__(self, op):
    device_string = ''
    if self._num_ps > 0:
      task_id = self._next_task_id
      self._next_task_id = (self._next_task_id + 1) % self._num_ps
      device_string = '%s/task:%d' % (self._ps_device, task_id)
    device_string += '/%s' % self._placement
    return device_string


# TODO(sguada) Remove once get_variable is able to colocate op.devices.
def variable_device(device, name):
  """Fix the variable device to colocate its ops."""
  if callable(device):
    var_name = tf.get_variable_scope().name + '/' + name
    var_def = tf.NodeDef(name=var_name, op='Variable')
    device = device(var_def)
  if device is None:
    device = ''
  return device


@scopes.add_arg_scope
def global_step(device=''):
  """Returns the global step variable.

  Args:
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.

  Returns:
    the tensor representing the global step variable.
  """
  global_step_ref = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
  if global_step_ref:
    return global_step_ref[0]
  else:
    collections = [
        VARIABLES_TO_RESTORE,
        tf.GraphKeys.VARIABLES,
        tf.GraphKeys.GLOBAL_STEP,
    ]
    # Get the device for the variable.
    with tf.device(variable_device(device, 'global_step')):
      return tf.get_variable('global_step', shape=[], dtype=tf.int64,
                             initializer=tf.zeros_initializer,
                             trainable=False, collections=collections)


@scopes.add_arg_scope
def variable(name, shape=None, dtype=tf.float32, initializer=None,
             regularizer=None, trainable=True, collections=None, device='',
             restore=True):
  """Gets an existing variable with these parameters or creates a new one.

    It also add itself to a group with its name.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of
        applying it on a newly created variable will be added to the collection
        GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the tf.GraphKeys.VARIABLES
      and MODEL_VARIABLES collections.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    restore: whether the variable should be added to the
      VARIABLES_TO_RESTORE collection.

  Returns:
    The created or existing variable.
  """
  collections = list(collections or [])

  # Make sure variables are added to tf.GraphKeys.VARIABLES and MODEL_VARIABLES
  collections += [tf.GraphKeys.VARIABLES, MODEL_VARIABLES]
  # Add to VARIABLES_TO_RESTORE if necessary
  if restore:
    collections.append(VARIABLES_TO_RESTORE)
  # Remove duplicates
  collections = set(collections)
  # Get the device for the variable.
  with tf.device(variable_device(device, name)):
    return tf.get_variable(name, shape=shape, dtype=dtype,
                           initializer=initializer, regularizer=regularizer,
                           trainable=trainable, collections=collections)
