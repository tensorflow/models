# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for params_dict.py."""

import os

import tensorflow as tf, tf_keras
import yaml

from official.modeling.hyperparams import params_dict


class ParamsDictTest(tf.test.TestCase):

  def test_init_from_an_empty_dict(self):
    params = params_dict.ParamsDict()
    with self.assertRaises(AttributeError):
      _ = params.a

    with self.assertRaises(KeyError):
      params.a = 'aa'

  def test_init_from_a_dict(self):
    params = params_dict.ParamsDict({'a': 'aa', 'b': 2})
    self.assertEqual(params.a, 'aa')
    self.assertEqual(params.b, 2)

  def test_init_from_a_param_dict(self):
    params_init = params_dict.ParamsDict({'a': 'aa', 'b': 2})
    params = params_dict.ParamsDict(params_init)
    self.assertEqual(params.a, 'aa')
    self.assertEqual(params.b, 2)

  def test_lock(self):
    params = params_dict.ParamsDict({'a': 1, 'b': 2, 'c': 3})
    params.lock()
    with self.assertRaises(ValueError):
      params.a = 10
    with self.assertRaises(ValueError):
      params.override({'b': 20})
    with self.assertRaises(ValueError):
      del params.c

  def test_setattr(self):
    params = params_dict.ParamsDict()
    params.override({'a': 'aa', 'b': 2, 'c': None}, is_strict=False)
    params.c = 'ccc'
    self.assertEqual(params.a, 'aa')
    self.assertEqual(params.b, 2)
    self.assertEqual(params.c, 'ccc')

  def test_getattr(self):
    params = params_dict.ParamsDict()
    params.override({'a': 'aa', 'b': 2, 'c': None}, is_strict=False)
    self.assertEqual(params.a, 'aa')
    self.assertEqual(params.b, 2)
    self.assertEqual(params.c, None)

  def test_delattr(self):
    params = params_dict.ParamsDict()
    params.override({
        'a': 'aa',
        'b': 2,
        'c': None,
        'd': {
            'd1': 1,
            'd2': 10
        }
    },
                    is_strict=False)
    del params.c
    self.assertEqual(params.a, 'aa')
    self.assertEqual(params.b, 2)
    with self.assertRaises(AttributeError):
      _ = params.c
    del params.d
    with self.assertRaises(AttributeError):
      _ = params.d.d1

  def test_contains(self):
    params = params_dict.ParamsDict()
    params.override({'a': 'aa'}, is_strict=False)
    self.assertIn('a', params)
    self.assertNotIn('b', params)

  def test_get(self):
    params = params_dict.ParamsDict()
    params.override({'a': 'aa'}, is_strict=False)
    self.assertEqual(params.get('a'), 'aa')
    self.assertEqual(params.get('b', 2), 2)
    self.assertEqual(params.get('b'), None)

  def test_override_is_strict_true(self):
    params = params_dict.ParamsDict({
        'a': 'aa',
        'b': 2,
        'c': {
            'c1': 'cc',
            'c2': 20
        }
    })
    params.override({'a': 2, 'c': {'c1': 'ccc'}}, is_strict=True)
    self.assertEqual(params.a, 2)
    self.assertEqual(params.c.c1, 'ccc')
    with self.assertRaises(KeyError):
      params.override({'d': 'ddd'}, is_strict=True)
    with self.assertRaises(KeyError):
      params.override({'c': {'c3': 30}}, is_strict=True)

  def test_override_is_strict_false(self):
    params = params_dict.ParamsDict({
        'a': 'aa',
        'b': 2,
        'c': {
            'c1': 10,
            'c2': 20
        }
    })
    params.override({'a': 2, 'c': {'c3': 3000}}, is_strict=False)
    self.assertEqual(params.a, 2)
    self.assertEqual(params.c.c3, 3000)
    params.override({'d': 'ddd'}, is_strict=False)
    self.assertEqual(params.d, 'ddd')
    params.override({'c': {'c4': 4444}}, is_strict=False)
    self.assertEqual(params.c.c4, 4444)

  def test_as_dict(self):
    params = params_dict.ParamsDict({
        'a': 'aa',
        'b': 2,
        'c': {
            'c1': 10,
            'c2': 20
        }
    })
    params_d = params.as_dict()
    self.assertEqual(params_d['a'], 'aa')
    self.assertEqual(params_d['b'], 2)
    self.assertEqual(params_d['c']['c1'], 10)
    self.assertEqual(params_d['c']['c2'], 20)

  def test_validate(self):
    # Raise error due to the unknown parameter.
    with self.assertRaises(KeyError):
      params = params_dict.ParamsDict({'a': 1, 'b': {'a': 11}}, ['a == c'])
      params.validate()

    # OK to check equality of two nested dicts.
    params = params_dict.ParamsDict({
        'a': 1,
        'b': {
            'a': 10
        },
        'c': {
            'a': 10
        }
    }, ['b == c'])
    params.validate()

    # Raise error due to inconsistency
    with self.assertRaises(KeyError):
      params = params_dict.ParamsDict({'a': 1, 'c': {'a': 10}}, ['a == c.a'])
      params.validate()

    # Valid rule.
    params = params_dict.ParamsDict({'a': 1, 'c': {'a': 1}}, ['a == c.a'])

    # Overriding violates the existing rule, raise error upon validate.
    params.override({'a': 11})
    with self.assertRaises(KeyError):
      params.validate()

    # Valid restrictions with constant.
    params = params_dict.ParamsDict({
        'a': None,
        'c': {
            'a': 1
        }
    }, ['a == None', 'c.a == 1'])
    params.validate()
    with self.assertRaises(KeyError):
      params = params_dict.ParamsDict({
          'a': 4,
          'c': {
              'a': 1
          }
      }, ['a == None', 'c.a == 1'])
      params.validate()

    # Valid restrictions with inequality.
    params = params_dict.ParamsDict({'a': 1}, ['a >= 1'])
    params.validate()


class ParamsDictIOTest(tf.test.TestCase):

  def write_temp_file(self, filename, text):
    temp_file = os.path.join(self.get_temp_dir(), filename)
    with tf.io.gfile.GFile(temp_file, 'w') as writer:
      writer.write(text)
    return temp_file

  def test_save_params_dict_to_yaml(self):
    params = params_dict.ParamsDict({
        'a': 'aa',
        'b': 2,
        'c': {
            'c1': 10,
            'c2': 20
        }
    })
    output_yaml_file = os.path.join(self.get_temp_dir(), 'params.yaml')
    params_dict.save_params_dict_to_yaml(params, output_yaml_file)

    with tf.io.gfile.GFile(output_yaml_file, 'r') as f:
      params_d = yaml.load(f, Loader=yaml.Loader)
      self.assertEqual(params.a, params_d['a'])
      self.assertEqual(params.b, params_d['b'])
      self.assertEqual(params.c.c1, params_d['c']['c1'])
      self.assertEqual(params.c.c2, params_d['c']['c2'])

  def test_read_yaml_to_params_dict(self):
    input_yaml_file = self.write_temp_file(
        'params.yaml', r"""
        a: 'aa'
        b: 2
        c:
          c1: 10
          c2: 20
    """)
    params = params_dict.read_yaml_to_params_dict(input_yaml_file)

    self.assertEqual(params.a, 'aa')
    self.assertEqual(params.b, 2)
    self.assertEqual(params.c.c1, 10)
    self.assertEqual(params.c.c2, 20)

  def test_override_params_dict_using_dict(self):
    params = params_dict.ParamsDict({
        'a': 1,
        'b': 2.5,
        'c': [3, 4],
        'd': 'hello',
        'e': False
    })
    override_dict = {'b': 5.2, 'c': [30, 40]}
    params = params_dict.override_params_dict(
        params, override_dict, is_strict=True)
    self.assertEqual(1, params.a)
    self.assertEqual(5.2, params.b)
    self.assertEqual([30, 40], params.c)
    self.assertEqual('hello', params.d)
    self.assertEqual(False, params.e)

  def test_override_params_dict_using_yaml_string(self):
    params = params_dict.ParamsDict({
        'a': 1,
        'b': 2.5,
        'c': [3, 4],
        'd': 'hello',
        'e': False
    })
    override_yaml_string = "'b': 5.2\n'c': [30, 40]"
    params = params_dict.override_params_dict(
        params, override_yaml_string, is_strict=True)
    self.assertEqual(1, params.a)
    self.assertEqual(5.2, params.b)
    self.assertEqual([30, 40], params.c)
    self.assertEqual('hello', params.d)
    self.assertEqual(False, params.e)

  def test_override_params_dict_using_json_string(self):
    params = params_dict.ParamsDict({
        'a': 1,
        'b': {
            'b1': 2,
            'b2': [2, 3],
        },
        'd': {
            'd1': {
                'd2': 'hello'
            }
        },
        'e': False
    })
    override_json_string = "{ b: { b2: [3, 4] }, d: { d1: { d2: 'hi' } } }"
    params = params_dict.override_params_dict(
        params, override_json_string, is_strict=True)
    self.assertEqual(1, params.a)
    self.assertEqual(2, params.b.b1)
    self.assertEqual([3, 4], params.b.b2)
    self.assertEqual('hi', params.d.d1.d2)
    self.assertEqual(False, params.e)

  def test_override_params_dict_using_csv_string(self):
    params = params_dict.ParamsDict({
        'a': 1,
        'b': {
            'b1': 2,
            'b2': [2, 3],
        },
        'd': {
            'd1': {
                'd2': 'hello'
            }
        },
        'e': False
    })
    override_csv_string = "b.b2=[3,4], d.d1.d2='hi, world', e=gs://test"
    params = params_dict.override_params_dict(
        params, override_csv_string, is_strict=True)
    self.assertEqual(1, params.a)
    self.assertEqual(2, params.b.b1)
    self.assertEqual([3, 4], params.b.b2)
    self.assertEqual('hi, world', params.d.d1.d2)
    self.assertEqual('gs://test', params.e)
    # Test different float formats
    override_csv_string = 'b.b2=-1.e-3, d.d1.d2=+0.001, e=1e+3, a=-1.5E-3'
    params = params_dict.override_params_dict(
        params, override_csv_string, is_strict=True)
    self.assertEqual(-1e-3, params.b.b2)
    self.assertEqual(0.001, params.d.d1.d2)
    self.assertEqual(1e3, params.e)
    self.assertEqual(-1.5e-3, params.a)

  def test_override_params_dict_using_yaml_file(self):
    params = params_dict.ParamsDict({
        'a': 1,
        'b': 2.5,
        'c': [3, 4],
        'd': 'hello',
        'e': False
    })
    override_yaml_file = self.write_temp_file(
        'params.yaml', r"""
        b: 5.2
        c: [30, 40]
        """)
    params = params_dict.override_params_dict(
        params, override_yaml_file, is_strict=True)
    self.assertEqual(1, params.a)
    self.assertEqual(5.2, params.b)
    self.assertEqual([30, 40], params.c)
    self.assertEqual('hello', params.d)
    self.assertEqual(False, params.e)


class IOTest(tf.test.TestCase):

  def test_basic_csv_str_to_json_str(self):
    csv_str = 'a=1,b=2,c=3'
    json_str = '{a : 1, b : 2, c : 3}'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    self.assertEqual(converted_csv_str, json_str)

  def test_basic_csv_str_load(self):
    csv_str = 'a=1,b=2,c=3'
    expected_output = {'a': 1, 'b': 2, 'c': 3}
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    converted_dict = yaml.load(converted_csv_str, Loader=yaml.Loader)
    self.assertDictEqual(converted_dict, expected_output)

  def test_basic_nested_csv_str_to_json_str(self):
    csv_str = 'a=1,b.b1=2'
    json_str = '{a : 1, b : {b1 : 2}}'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    self.assertEqual(converted_csv_str, json_str)

  def test_basic_nested_csv_str_load(self):
    csv_str = 'a=1,b.b1=2,c.c1=3'
    expected_output = {'a': 1, 'b': {'b1': 2}, 'c': {'c1': 3}}
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    converted_dict = yaml.load(converted_csv_str, Loader=yaml.Loader)
    self.assertDictEqual(converted_dict, expected_output)

  def test_complex_nested_csv_str_to_json_str(self):
    csv_str = 'a.aa.aaa.aaaaa.a=1'
    json_str = '{a : {aa : {aaa : {aaaaa : {a : 1}}}}}'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    self.assertEqual(converted_csv_str, json_str)

  def test_complex_nested_csv_str_load(self):
    csv_str = 'a.aa.aaa.aaaaa.a=1,a.a=2'
    expected_output = {'a': {'aa': {'aaa': {'aaaaa': {'a': 1}}}, 'a': 2}}
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    converted_dict = yaml.load(converted_csv_str, Loader=yaml.Loader)
    self.assertDictEqual(converted_dict, expected_output)

  def test_int_array_param_nested_csv_str_to_json_str(self):
    csv_str = 'a.b[2]=3,a.b[0]=1,a.b[1]=2'
    json_str = '{a : {b : [1, 2, 3]}}'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    self.assertEqual(converted_csv_str, json_str)

  def test_float_array_param_nested_csv_str_to_json_str(self):
    csv_str = 'a.b[1]=3.45,a.b[2]=1.32,a.b[0]=2.232'
    json_str = '{a : {b : [2.232, 3.45, 1.32]}}'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    self.assertEqual(converted_csv_str, json_str)

  def test_incomplete_array_param_nested_csv_str_to_json_str(self):
    csv_str = 'a.b[0]=1,a.b[2]=2'
    self.assertRaises(ValueError, params_dict.nested_csv_str_to_json_str,
                      csv_str)

  def test_csv_str_load_supported_datatypes(self):
    csv_str = 'a=1,b=2.,c=[1,2,3],d=\'hello, there\',e=\"Hi.\"'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    converted_dict = yaml.load(converted_csv_str, Loader=yaml.Loader)
    self.assertEqual(converted_dict['a'], 1)
    self.assertEqual(converted_dict['b'], 2.)
    self.assertEqual(converted_dict['c'], [1, 2, 3])
    self.assertEqual(converted_dict['d'], 'hello, there')
    self.assertEqual(converted_dict['e'], 'Hi.')

  def test_csv_str_load_unsupported_datatypes(self):
    csv_str = 'a=[[1,2,3],[4,5,6]]'
    self.assertRaises(ValueError, params_dict.nested_csv_str_to_json_str,
                      csv_str)

  def test_csv_str_to_json_str_spacing(self):
    csv_str1 = 'a=1,b=2,c=3'
    csv_str2 = 'a = 1, b = 2, c = 3'
    json_str = '{a : 1, b : 2, c : 3}'
    converted_csv_str1 = params_dict.nested_csv_str_to_json_str(csv_str1)
    converted_csv_str2 = params_dict.nested_csv_str_to_json_str(csv_str2)
    self.assertEqual(converted_csv_str1, converted_csv_str2)
    self.assertEqual(converted_csv_str1, json_str)
    self.assertEqual(converted_csv_str2, json_str)

  def test_gcs_added_quotes(self):
    csv_str = 'a=gs://abc, b=gs://def'
    expected_output = '{a : \'gs://abc\', b : \'gs://def\'}'
    converted_csv_str = params_dict.nested_csv_str_to_json_str(csv_str)
    self.assertEqual(converted_csv_str, expected_output)


if __name__ == '__main__':
  tf.test.main()
