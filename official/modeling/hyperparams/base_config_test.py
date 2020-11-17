# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import pprint
from typing import List, Tuple

from absl.testing import parameterized
import dataclasses
import tensorflow as tf
from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class DumpConfig1(base_config.Config):
  a: int = 1
  b: str = 'text'


@dataclasses.dataclass
class DumpConfig2(base_config.Config):
  c: int = 2
  d: str = 'text'
  e: DumpConfig1 = DumpConfig1()


@dataclasses.dataclass
class DumpConfig3(DumpConfig2):
  f: int = 2
  g: str = 'text'
  h: List[DumpConfig1] = dataclasses.field(
      default_factory=lambda: [DumpConfig1(), DumpConfig1()])
  g: Tuple[DumpConfig1, ...] = (DumpConfig1(),)


@dataclasses.dataclass
class DumpConfig4(DumpConfig2):
  x: int = 3


@dataclasses.dataclass
class DummyConfig5(base_config.Config):
  y: Tuple[DumpConfig2, ...] = (DumpConfig2(), DumpConfig4())
  z: Tuple[str] = ('a',)


class BaseConfigTest(parameterized.TestCase, tf.test.TestCase):

  def assertHasSameTypes(self, c, d, msg=''):
    """Checks if a Config has the same structure as a given dict.

    Args:
      c: the Config object to be check.
      d: the reference dict object.
      msg: The error message to show when type mismatched.
    """
    # Make sure d is not a Config. Assume d is either
    # dictionary or primitive type and c is the Config or primitive types.
    self.assertNotIsInstance(d, base_config.Config)
    if isinstance(d, base_config.Config.IMMUTABLE_TYPES):
      self.assertEqual(pprint.pformat(c), pprint.pformat(d), msg=msg)
    elif isinstance(d, base_config.Config.SEQUENCE_TYPES):
      self.assertEqual(type(c), type(d), msg=msg)
      for i, v in enumerate(d):
        self.assertHasSameTypes(c[i], v, msg='{}[{!r}]'.format(msg, i))
    elif isinstance(d, dict):
      self.assertIsInstance(c, base_config.Config, msg=msg)
      for k, v in sorted(d.items()):
        self.assertHasSameTypes(getattr(c, k), v, msg='{}[{!r}]'.format(msg, k))
    else:
      raise TypeError('Unknown type: %r' % type(d))

  def assertImportExport(self, v):
    config = base_config.Config({'key': v})
    back = config.as_dict()['key']
    self.assertEqual(pprint.pformat(back), pprint.pformat(v))
    self.assertHasSameTypes(config.key, v, msg='=%s v' % pprint.pformat(v))

  def test_invalid_keys(self):
    params = base_config.Config()
    with self.assertRaises(AttributeError):
      _ = params.a

  def test_nested_config_types(self):
    config = DumpConfig3()
    self.assertIsInstance(config.e, DumpConfig1)
    self.assertIsInstance(config.h[0], DumpConfig1)
    self.assertIsInstance(config.h[1], DumpConfig1)
    self.assertIsInstance(config.g[0], DumpConfig1)

    config.override({'e': {'a': 2, 'b': 'new text'}})
    self.assertIsInstance(config.e, DumpConfig1)
    self.assertEqual(config.e.a, 2)
    self.assertEqual(config.e.b, 'new text')

    config.override({'h': [{'a': 3, 'b': 'new text 2'}]})
    self.assertIsInstance(config.h[0], DumpConfig1)
    self.assertLen(config.h, 1)
    self.assertEqual(config.h[0].a, 3)
    self.assertEqual(config.h[0].b, 'new text 2')

    config.override({'g': [{'a': 4, 'b': 'new text 3'}]})
    self.assertIsInstance(config.g[0], DumpConfig1)
    self.assertLen(config.g, 1)
    self.assertEqual(config.g[0].a, 4)
    self.assertEqual(config.g[0].b, 'new text 3')

  def test_replace(self):
    config = DumpConfig2()
    new_config = config.replace(e={'a': 2})
    self.assertEqual(new_config.e.a, 2)
    self.assertIsInstance(new_config.e, DumpConfig1)

    config = DumpConfig2(e=DumpConfig2())
    new_config = config.replace(e={'c': 4})
    self.assertEqual(new_config.e.c, 4)
    self.assertIsInstance(new_config.e, DumpConfig2)

    config = DumpConfig3()
    new_config = config.replace(g=[{'a': 4, 'b': 'new text 3'}])
    self.assertIsInstance(new_config.g[0], DumpConfig1)
    self.assertEqual(new_config.g[0].a, 4)

  @parameterized.parameters(
      ('_locked', "The key '_locked' is internally reserved."),
      ('_restrictions', "The key '_restrictions' is internally reserved."),
      ('aa', "The key 'aa' does not exist."),
  )
  def test_key_error(self, key, msg):
    params = base_config.Config()
    with self.assertRaisesRegex(KeyError, msg):
      params.override({key: True})

  @parameterized.parameters(
      ('str data',),
      (123,),
      (1.23,),
      (None,),
      (['str', 1, 2.3, None],),
      (('str', 1, 2.3, None),),
  )
  def test_import_export_immutable_types(self, v):
    self.assertImportExport(v)
    out = base_config.Config({'key': v})
    self.assertEqual(pprint.pformat(v), pprint.pformat(out.key))

  def test_override_is_strict_true(self):
    params = base_config.Config({
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

    config = base_config.Config({'key': [{'a': 42}]})
    with self.assertRaisesRegex(KeyError, "The key 'b' does not exist"):
      config.override({'key': [{'b': 43}]})

  @parameterized.parameters(
      (lambda x: x, 'Unknown type'),
      (object(), 'Unknown type'),
      (set(), 'Unknown type'),
      (frozenset(), 'Unknown type'),
  )
  def test_import_unsupport_types(self, v, msg):
    with self.assertRaisesRegex(TypeError, msg):
      _ = base_config.Config({'key': v})

  @parameterized.parameters(
      ({
          'a': [{
              'b': 2,
          }, {
              'c': 3,
          }]
      },),
      ({
          'c': [{
              'f': 1.1,
          }, {
              'h': [1, 2],
          }]
      },),
      (({
          'a': 'aa',
          'b': 2,
          'c': {
              'c1': 10,
              'c2': 20,
          }
      },),),
  )
  def test_import_export_nested_structure(self, d):
    self.assertImportExport(d)

  @parameterized.parameters(
      ([{
          'a': 42,
          'b': 'hello',
          'c': 1.2
      }],),
      (({
          'a': 42,
          'b': 'hello',
          'c': 1.2
      },),),
  )
  def test_import_export_nested_sequences(self, v):
    self.assertImportExport(v)

  @parameterized.parameters(
      ([([{}],)],),
      ([['str', 1, 2.3, None]],),
      ((('str', 1, 2.3, None),),),
      ([
          ('str', 1, 2.3, None),
      ],),
      ([
          ('str', 1, 2.3, None),
      ],),
      ([[{
          'a': 42,
          'b': 'hello',
          'c': 1.2
      }]],),
      ([[[{
          'a': 42,
          'b': 'hello',
          'c': 1.2
      }]]],),
      ((({
          'a': 42,
          'b': 'hello',
          'c': 1.2
      },),),),
      (((({
          'a': 42,
          'b': 'hello',
          'c': 1.2
      },),),),),
      ([({
          'a': 42,
          'b': 'hello',
          'c': 1.2
      },)],),
      (([{
          'a': 42,
          'b': 'hello',
          'c': 1.2
      }],),),
  )
  def test_import_export_unsupport_sequence(self, v):
    with self.assertRaisesRegex(TypeError,
                                'Invalid sequence: only supports single level'):
      _ = base_config.Config({'key': v})

  def test_construct_subtype(self):
    pass

  def test_import_config(self):
    params = base_config.Config({'a': [{'b': 2}, {'c': {'d': 3}}]})
    self.assertLen(params.a, 2)
    self.assertEqual(params.a[0].b, 2)
    self.assertEqual(type(params.a[0]), base_config.Config)
    self.assertEqual(pprint.pformat(params.a[0].b), '2')
    self.assertEqual(type(params.a[1]), base_config.Config)
    self.assertEqual(type(params.a[1].c), base_config.Config)
    self.assertEqual(pprint.pformat(params.a[1].c.d), '3')

  def test_override(self):
    params = base_config.Config({'a': [{'b': 2}, {'c': {'d': 3}}]})
    params.override({'a': [{'b': 4}, {'c': {'d': 5}}]}, is_strict=False)
    self.assertEqual(type(params.a), list)
    self.assertEqual(type(params.a[0]), base_config.Config)
    self.assertEqual(pprint.pformat(params.a[0].b), '4')
    self.assertEqual(type(params.a[1]), base_config.Config)
    self.assertEqual(type(params.a[1].c), base_config.Config)
    self.assertEqual(pprint.pformat(params.a[1].c.d), '5')

  @parameterized.parameters(
      ([{}],),
      (({},),),
  )
  def test_config_vs_params_dict(self, v):
    d = {'key': v}
    self.assertEqual(type(base_config.Config(d).key[0]), base_config.Config)
    self.assertEqual(type(base_config.params_dict.ParamsDict(d).key[0]), dict)

  def test_ppformat(self):
    self.assertEqual(
        pprint.pformat([
            's', 1, 1.0, True, None, {}, [], (), {
                (2,): (3, [4], {
                    6: 7,
                }),
                8: 9,
            }
        ]),
        "['s', 1, 1.0, True, None, {}, [], (), {8: 9, (2,): (3, [4], {6: 7})}]")

  def test_with_restrictions(self):
    restrictions = ['e.a<c']
    config = DumpConfig2(restrictions=restrictions)
    config.validate()

  def test_nested_tuple(self):
    config = DummyConfig5()
    config.override({
        'y': [{
            'c': 4,
            'd': 'new text 3',
            'e': {
                'a': 2
            }
        }, {
            'c': 0,
            'd': 'new text 3',
            'e': {
                'a': 2
            }
        }],
        'z': ['a', 'b', 'c'],
    })
    self.assertEqual(config.y[0].c, 4)
    self.assertEqual(config.y[1].c, 0)
    self.assertIsInstance(config.y[0], DumpConfig2)
    self.assertIsInstance(config.y[1], DumpConfig4)
    self.assertSameElements(config.z, ['a', 'b', 'c'])

  def test_override_by_empty_sequence(self):
    config = DummyConfig5()
    config.override({
        'y': [],
        'z': (),
    }, is_strict=True)
    self.assertEmpty(config.y)
    self.assertEmpty(config.z)


if __name__ == '__main__':
  tf.test.main()
