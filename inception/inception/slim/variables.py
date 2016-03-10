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
"""Contains convenience wrappers for creating Variables in TensorFlow.

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
  with slim.arg_scope(variables.Variables, restore=False):
    net = slim.ops.conv2d(net, 64, [3, 3], scope='conv3')

  # Get all model variables from all the layers.
  model_variables = slim.variables.get_variables()

  # Get all model variables from a specific the layer, i.e 'conv1'.
  conv1_variables = slim.variables.get_variables('conv1')

  # Get all weights from all the layers.
  weights = slim.variables.get_variables_by_name('weights')

  # Get all bias from all the layers.
  biases = slim.variables.get_variables_by_name('biases')

  # Get all variables in the VARIABLES_TO_RESTORE collection
  # (i.e. only those created by 'conv1' and 'conv2')
  variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)

************************************************
* Initializing model variables from a checkpoint
************************************************

# Create some variables.
v1 = slim.variables.variable(name="v1", ..., restore=False)
v2 = slim.variables.variable(name="v2", ...) # By default restore=True
...
# The list of variables to restore should only contain 'v2'.
variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
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
VARIABLES_COLLECTION = '_variables_'

# Collection containing all the slim.variables that are marked to_restore
VARIABLES_TO_RESTORE = '_variables_to_restore_'


def get_variable_given_name(var):
  """Gets the variable given name without the scope.

  Args:
    var: a variable.

  Returns:
    the given name of the variable without the scope.
  """
  name = var.op.name
  if '/' in name:
    name = name.split('/')[-1]
  return name


def default_collections(given_name, restore):
  """Define the set of default collections that variables should be added.

  Args:
    given_name: the given name of the variable.
    restore: whether the variable should be added to the VARIABLES_TO_RESTORE
      collection.

  Returns:
    a list of default collections.
  """
  defaults = [tf.GraphKeys.VARIABLES, VARIABLES_COLLECTION]
  defaults += [VARIABLES_COLLECTION + given_name]
  if restore:
    defaults += [VARIABLES_TO_RESTORE]
  return defaults


def add_variable(var, restore=True):
  """Adds a variable to the default set of collections.

  Args:
    var: a variable.
    restore: whether the variable should be added to the
      VARIABLES_TO_RESTORE collection.
  """
  given_name = get_variable_given_name(var)
  for collection in default_collections(given_name, restore):
    if var not in tf.get_collection(collection):
      tf.add_to_collection(collection, var)


def get_variables(prefix=None, suffix=None):
  """Gets the list of variables, filtered by prefix and/or suffix.

  Args:
    prefix: an optional prefix for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables with prefix and suffix.
  """
  candidates = tf.get_collection(VARIABLES_COLLECTION, prefix)
  if suffix is not None:
    candidates = [var for var in candidates if var.op.name.endswith(suffix)]
  return candidates


def get_variables_by_name(given_name, prefix=None):
  """Gets the list of variables were given that name.

  Args:
    given_name: name given to the variable without scope.
    prefix: an optional prefix for filtering the variables to return.

  Returns:
    a list of variables with prefix and suffix.
  """
  return tf.get_collection(VARIABLES_COLLECTION + given_name, prefix)


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
      collection.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    restore: whether the variable should be added to the
      VARIABLES_TO_RESTORE collection.

  Returns:
    The created or existing variable.
  """
  # Instantiate the device for this variable if it is passed as a function.
  if device and callable(device):
    device = device()
  collections = set(list(collections or []) + default_collections(name,
                                                                  restore))
  with tf.device(device):
    return tf.get_variable(name, shape=shape, dtype=dtype,
                           initializer=initializer, regularizer=regularizer,
                           trainable=trainable, collections=collections)
