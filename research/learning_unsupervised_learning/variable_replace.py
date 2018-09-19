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

import tensorflow as tf
from contextlib import contextmanager

from tensorflow.python.ops import variable_scope

# sanity global state to ensure non recursive.
_is_variable_replacing = [False]

def in_variable_replace_scope():
  return _is_variable_replacing[0]

@contextmanager
def variable_replace(replacements, no_new=True):
  """ A context manager that replaces variables.

  This is a context manager that replaces all calls to
  get_variable with the variable in replacements.
  This function does not support recursive application.

  Args:
    replacements: dict
        dictionary mapping a variable to replace (the key), with
        the variable one wants to replace this variable with (the value).
    no_new: bool
        raise an error if variables were created.
        This is for sanity checking.
  Raises:
    ValueError: if a new variable or not all the replacements are used.
  """
  # TODO(lmetz) This function is a bit scary, as it relies on monkey patching
  # the call to get_variable. Ideally this can be done with variable_scope's
  # custom_getter attribute, but when initially writing this that was not
  # avalible.

  replacements = {k: v for k, v in replacements.items() if not k == v}

  init_vars = tf.trainable_variables()
  old_get_variable = variable_scope.get_variable
  old_tf_get_variable = tf.get_variable

  names_replace = {}
  has_replaced_names = []
  tf.logging.vlog(2, "Trying to replace")
  for k, v in replacements.items():
    tf.logging.vlog(2, k.name + " >> " + v.name)
  tf.logging.vlog(2, "===")

  for k, v in replacements.items():
    strip_name = k.name.replace("/read:0", "")
    strip_name = strip_name.replace(":0", "")
    names_replace[strip_name] = v
    # TODO(lmetz) is there a cleaner way to do this?
  def new_get_variable(name, *args, **kwargs):
    #print "Monkeypatch get variable run with name:", name
    n = tf.get_variable_scope().name + "/" + name
    #print "Monkeypatch get variable run with name:", n
    if n in names_replace:
      has_replaced_names.append(n)
      return names_replace[n]
    else:
      return old_get_variable(name, *args, **kwargs)

  # perform the monkey patch
  if _is_variable_replacing[0] == True:
    raise ValueError("No recursive calling to variable replace allowed.")

  variable_scope.get_variable = new_get_variable
  tf.get_variable = new_get_variable

  _is_variable_replacing[0] = True

  yield

  if set(has_replaced_names) != set(names_replace.keys()):
    print "Didn't use all replacements"
    print "replaced variables that are not requested??"
    print "==="
    for n in list(set(has_replaced_names) - set(names_replace.keys())):
      print n
    print "Missed replacing variables"
    print "==="
    for n in list(set(names_replace.keys()) - set(has_replaced_names)):
      print n, "==>", names_replace[n].name
    raise ValueError("Fix this -- see stderr")

  # undo the monkey patch
  tf.get_variable = old_tf_get_variable
  variable_scope.get_variable = old_get_variable

  _is_variable_replacing[0] = False

  final_vars = tf.trainable_variables()
  assert set(init_vars) == set(final_vars), "trainable variables changed"
