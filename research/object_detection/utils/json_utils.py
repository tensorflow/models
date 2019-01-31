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
"""Utilities for dealing with writing json strings.

json_utils wraps json.dump and json.dumps so that they can be used to safely
control the precision of floats when writing to json strings or files.
"""
import json
from json import encoder


def Dump(obj, fid, float_digits=-1, **params):
  """Wrapper of json.dump that allows specifying the float precision used.

  Args:
    obj: The object to dump.
    fid: The file id to write to.
    float_digits: The number of digits of precision when writing floats out.
    **params: Additional parameters to pass to json.dumps.
  """
  original_encoder = encoder.FLOAT_REPR
  if float_digits >= 0:
    encoder.FLOAT_REPR = lambda o: format(o, '.%df' % float_digits)
  try:
    json.dump(obj, fid, **params)
  finally:
    encoder.FLOAT_REPR = original_encoder


def Dumps(obj, float_digits=-1, **params):
  """Wrapper of json.dumps that allows specifying the float precision used.

  Args:
    obj: The object to dump.
    float_digits: The number of digits of precision when writing floats out.
    **params: Additional parameters to pass to json.dumps.

  Returns:
    output: JSON string representation of obj.
  """
  original_encoder = encoder.FLOAT_REPR
  original_c_make_encoder = encoder.c_make_encoder
  if float_digits >= 0:
    encoder.FLOAT_REPR = lambda o: format(o, '.%df' % float_digits)
    encoder.c_make_encoder = None
  try:
    output = json.dumps(obj, **params)
  finally:
    encoder.FLOAT_REPR = original_encoder
    encoder.c_make_encoder = original_c_make_encoder

  return output


def PrettyParams(**params):
  """Returns parameters for use with Dump and Dumps to output pretty json.

  Example usage:
    ```json_str = json_utils.Dumps(obj, **json_utils.PrettyParams())```
    ```json_str = json_utils.Dumps(
                      obj, **json_utils.PrettyParams(allow_nans=False))```

  Args:
    **params: Additional params to pass to json.dump or json.dumps.

  Returns:
    params: Parameters that are compatible with json_utils.Dump and
      json_utils.Dumps.
  """
  params['float_digits'] = 4
  params['sort_keys'] = True
  params['indent'] = 2
  params['separators'] = (',', ': ')
  return params

