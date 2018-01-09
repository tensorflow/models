from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.config_lib."""

import tensorflow as tf

from common import config_lib  # brain coder


class ConfigLibTest(tf.test.TestCase):

  def testConfig(self):
    config = config_lib.Config(hello='world', foo='bar', num=123, f=56.7)
    self.assertEqual('world', config.hello)
    self.assertEqual('bar', config['foo'])
    config.hello = 'everyone'
    config['bar'] = 9000
    self.assertEqual('everyone', config['hello'])
    self.assertEqual(9000, config.bar)
    self.assertEqual(5, len(config))

  def testConfigUpdate(self):
    config = config_lib.Config(a=1, b=2, c=3)
    config.update({'b': 10, 'd': 4})
    self.assertEqual({'a': 1, 'b': 10, 'c': 3, 'd': 4}, config)

    config = config_lib.Config(a=1, b=2, c=3)
    config.update(b=10, d=4)
    self.assertEqual({'a': 1, 'b': 10, 'c': 3, 'd': 4}, config)

    config = config_lib.Config(a=1, b=2, c=3)
    config.update({'e': 5}, b=10, d=4)
    self.assertEqual({'a': 1, 'b': 10, 'c': 3, 'd': 4, 'e': 5}, config)

    config = config_lib.Config(
        a=1,
        b=2,
        x=config_lib.Config(
            l='a',
            y=config_lib.Config(m=1, n=2),
            z=config_lib.Config(
                q=config_lib.Config(a=10, b=20),
                r=config_lib.Config(s=1, t=2))))
    config.update(x={'y': {'m': 10}, 'z': {'r': {'s': 5}}})
    self.assertEqual(
        config_lib.Config(
            a=1, b=2,
            x=config_lib.Config(
                l='a',
                y=config_lib.Config(m=10, n=2),
                z=config_lib.Config(
                    q=config_lib.Config(a=10, b=20),
                    r=config_lib.Config(s=5, t=2)))),
        config)

    config = config_lib.Config(
        foo='bar',
        num=100,
        x=config_lib.Config(a=1, b=2, c=config_lib.Config(h=10, i=20, j=30)),
        y=config_lib.Config(qrs=5, tuv=10),
        d={'a': 1, 'b': 2},
        l=[1, 2, 3])
    config.update(
        config_lib.Config(
            foo='hat',
            num=50.5,
            x={'a': 5, 'z': -10},
            y=config_lib.Config(wxyz=-1)),
        d={'a': 10, 'c': 20},
        l=[3, 4, 5, 6])
    self.assertEqual(
        config_lib.Config(
            foo='hat',
            num=50.5,
            x=config_lib.Config(a=5, b=2, z=-10,
                                c=config_lib.Config(h=10, i=20, j=30)),
            y=config_lib.Config(qrs=5, tuv=10, wxyz=-1),
            d={'a': 10, 'c': 20},
            l=[3, 4, 5, 6]),
        config)
    self.assertTrue(isinstance(config.x, config_lib.Config))
    self.assertTrue(isinstance(config.x.c, config_lib.Config))
    self.assertTrue(isinstance(config.y, config_lib.Config))

    config = config_lib.Config(
        foo='bar',
        num=100,
        x=config_lib.Config(a=1, b=2, c=config_lib.Config(h=10, i=20, j=30)),
        y=config_lib.Config(qrs=5, tuv=10),
        d={'a': 1, 'b': 2},
        l=[1, 2, 3])
    config.update(
        config_lib.Config(
            foo=1234,
            num='hello',
            x={'a': 5, 'z': -10, 'c': {'h': -5, 'k': 40}},
            y=[1, 2, 3, 4],
            d='stuff',
            l={'a': 1, 'b': 2}))
    self.assertEqual(
        config_lib.Config(
            foo=1234,
            num='hello',
            x=config_lib.Config(a=5, b=2, z=-10,
                                c=config_lib.Config(h=-5, i=20, j=30, k=40)),
            y=[1, 2, 3, 4],
            d='stuff',
            l={'a': 1, 'b': 2}),
        config)
    self.assertTrue(isinstance(config.x, config_lib.Config))
    self.assertTrue(isinstance(config.x.c, config_lib.Config))
    self.assertTrue(isinstance(config.y, list))

  def testConfigStrictUpdate(self):
    config = config_lib.Config(a=1, b=2, c=3)
    config.strict_update({'b': 10, 'c': 20})
    self.assertEqual({'a': 1, 'b': 10, 'c': 20}, config)

    config = config_lib.Config(a=1, b=2, c=3)
    config.strict_update(b=10, c=20)
    self.assertEqual({'a': 1, 'b': 10, 'c': 20}, config)

    config = config_lib.Config(a=1, b=2, c=3, d=4)
    config.strict_update({'d': 100}, b=10, a=20)
    self.assertEqual({'a': 20, 'b': 10, 'c': 3, 'd': 100}, config)

    config = config_lib.Config(
        a=1,
        b=2,
        x=config_lib.Config(
            l='a',
            y=config_lib.Config(m=1, n=2),
            z=config_lib.Config(
                q=config_lib.Config(a=10, b=20),
                r=config_lib.Config(s=1, t=2))))
    config.strict_update(x={'y': {'m': 10}, 'z': {'r': {'s': 5}}})
    self.assertEqual(
        config_lib.Config(
            a=1, b=2,
            x=config_lib.Config(
                l='a',
                y=config_lib.Config(m=10, n=2),
                z=config_lib.Config(
                    q=config_lib.Config(a=10, b=20),
                    r=config_lib.Config(s=5, t=2)))),
        config)

    config = config_lib.Config(
        foo='bar',
        num=100,
        x=config_lib.Config(a=1, b=2, c=config_lib.Config(h=10, i=20, j=30)),
        y=config_lib.Config(qrs=5, tuv=10),
        d={'a': 1, 'b': 2},
        l=[1, 2, 3])
    config.strict_update(
        config_lib.Config(
            foo='hat',
            num=50,
            x={'a': 5, 'c': {'h': 100}},
            y=config_lib.Config(tuv=-1)),
        d={'a': 10, 'c': 20},
        l=[3, 4, 5, 6])
    self.assertEqual(
        config_lib.Config(
            foo='hat',
            num=50,
            x=config_lib.Config(a=5, b=2,
                                c=config_lib.Config(h=100, i=20, j=30)),
            y=config_lib.Config(qrs=5, tuv=-1),
            d={'a': 10, 'c': 20},
            l=[3, 4, 5, 6]),
        config)

  def testConfigStrictUpdateFail(self):
    config = config_lib.Config(a=1, b=2, c=3, x=config_lib.Config(a=1, b=2))
    with self.assertRaises(KeyError):
      config.strict_update({'b': 10, 'c': 20, 'd': 50})
    with self.assertRaises(KeyError):
      config.strict_update(b=10, d=50)
    with self.assertRaises(KeyError):
      config.strict_update(x={'c': 3})
    with self.assertRaises(TypeError):
      config.strict_update(a='string')
    with self.assertRaises(TypeError):
      config.strict_update(x={'a': 'string'})
    with self.assertRaises(TypeError):
      config.strict_update(x=[1, 2, 3])

  def testConfigFromStr(self):
    config = config_lib.Config.from_str("{'c': {'d': 5}, 'b': 2, 'a': 1}")
    self.assertEqual(
        {'c': {'d': 5}, 'b': 2, 'a': 1}, config)
    self.assertTrue(isinstance(config, config_lib.Config))
    self.assertTrue(isinstance(config.c, config_lib.Config))

  def testConfigParse(self):
    config = config_lib.Config.parse(
        'hello="world",num=1234.5,lst=[10,20.5,True,"hi",("a","b","c")],'
        'dct={9:10,"stuff":"qwerty","subdict":{1:True,2:False}},'
        'subconfig=c(a=1,b=[1,2,[3,4]],c=c(f="f",g="g"))')
    self.assertEqual(
        {'hello': 'world', 'num': 1234.5,
         'lst': [10, 20.5, True, 'hi', ('a', 'b', 'c')],
         'dct': {9: 10, 'stuff': 'qwerty', 'subdict': {1: True, 2: False}},
         'subconfig': {'a': 1, 'b': [1, 2, [3, 4]], 'c': {'f': 'f', 'g': 'g'}}},
        config)
    self.assertTrue(isinstance(config, config_lib.Config))
    self.assertTrue(isinstance(config.subconfig, config_lib.Config))
    self.assertTrue(isinstance(config.subconfig.c, config_lib.Config))
    self.assertFalse(isinstance(config.dct, config_lib.Config))
    self.assertFalse(isinstance(config.dct['subdict'], config_lib.Config))
    self.assertTrue(isinstance(config.lst[4], tuple))

  def testConfigParseErrors(self):
    with self.assertRaises(SyntaxError):
      config_lib.Config.parse('a=[1,2,b="hello"')
    with self.assertRaises(SyntaxError):
      config_lib.Config.parse('a=1,b=c(x="a",y="b"')
    with self.assertRaises(SyntaxError):
      config_lib.Config.parse('a=1,b=c(x="a")y="b"')
    with self.assertRaises(SyntaxError):
      config_lib.Config.parse('a=1,b=c(x="a"),y="b",')

  def testOneOf(self):
    def make_config():
      return config_lib.Config(
          data=config_lib.OneOf(
              [config_lib.Config(task=1, a='hello'),
               config_lib.Config(task=2, a='world', b='stuff'),
               config_lib.Config(task=3, c=1234)],
              task=2),
          model=config_lib.Config(stuff=1))

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=1,a="hi")'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=1, a='hi'),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=2,a="hi")'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=2, a='hi', b='stuff'),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=3)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=3, c=1234),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=2, a='world', b='stuff'),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=4,d=9999)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=4, d=9999),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2),data=5'))
    self.assertEqual(
        config_lib.Config(
            data=5,
            model=config_lib.Config(stuff=2)),
        config)

  def testOneOfStrict(self):
    def make_config():
      return config_lib.Config(
          data=config_lib.OneOf(
              [config_lib.Config(task=1, a='hello'),
               config_lib.Config(task=2, a='world', b='stuff'),
               config_lib.Config(task=3, c=1234)],
              task=2),
          model=config_lib.Config(stuff=1))

    config = make_config()
    config.strict_update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=1,a="hi")'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=1, a='hi'),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.strict_update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=2,a="hi")'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=2, a='hi', b='stuff'),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.strict_update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=3)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=3, c=1234),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.strict_update(config_lib.Config.parse(
        'model=c(stuff=2)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(task=2, a='world', b='stuff'),
            model=config_lib.Config(stuff=2)),
        config)

  def testNestedOneOf(self):
    def make_config():
      return config_lib.Config(
          data=config_lib.OneOf(
              [config_lib.Config(task=1, a='hello'),
               config_lib.Config(
                   task=2,
                   a=config_lib.OneOf(
                       [config_lib.Config(x=1, y=2),
                        config_lib.Config(x=-1, y=1000, z=4)],
                       x=1)),
               config_lib.Config(task=3, c=1234)],
              task=2),
          model=config_lib.Config(stuff=1))

    config = make_config()
    config.update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=2,a=c(x=-1,z=8))'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(
                task=2,
                a=config_lib.Config(x=-1, y=1000, z=8)),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.strict_update(config_lib.Config.parse(
        'model=c(stuff=2),data=c(task=2,a=c(x=-1,z=8))'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(
                task=2,
                a=config_lib.Config(x=-1, y=1000, z=8)),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.update(config_lib.Config.parse('model=c(stuff=2)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(
                task=2,
                a=config_lib.Config(x=1, y=2)),
            model=config_lib.Config(stuff=2)),
        config)

    config = make_config()
    config.strict_update(config_lib.Config.parse('model=c(stuff=2)'))
    self.assertEqual(
        config_lib.Config(
            data=config_lib.Config(
                task=2,
                a=config_lib.Config(x=1, y=2)),
            model=config_lib.Config(stuff=2)),
        config)

  def testOneOfStrictErrors(self):
    def make_config():
      return config_lib.Config(
          data=config_lib.OneOf(
              [config_lib.Config(task=1, a='hello'),
               config_lib.Config(task=2, a='world', b='stuff'),
               config_lib.Config(task=3, c=1234)],
              task=2),
          model=config_lib.Config(stuff=1))

    config = make_config()
    with self.assertRaises(TypeError):
      config.strict_update(config_lib.Config.parse(
          'model=c(stuff=2),data=[1,2,3]'))

    config = make_config()
    with self.assertRaises(KeyError):
      config.strict_update(config_lib.Config.parse(
          'model=c(stuff=2),data=c(task=3,c=5678,d=9999)'))

    config = make_config()
    with self.assertRaises(ValueError):
      config.strict_update(config_lib.Config.parse(
          'model=c(stuff=2),data=c(task=4,d=9999)'))

    config = make_config()
    with self.assertRaises(TypeError):
      config.strict_update(config_lib.Config.parse(
          'model=c(stuff=2),data=5'))


if __name__ == '__main__':
  tf.test.main()
