"""Tests for check assertions."""

from tensorflow.python.platform import googletest

from syntaxnet.util import check


class RegistryTest(googletest.TestCase):
  """Testing rig."""

  def testCheckEq(self):
    check.Eq(1, 1, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Eq(1, 2, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Eq(1, 2, 'baz', RuntimeError)

  def testCheckNe(self):
    check.Ne(1, 2, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Ne(1, 1, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Ne(1, 1, 'baz', RuntimeError)

  def testCheckLt(self):
    check.Lt(1, 2, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Lt(1, 1, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Lt(1, -1, 'baz', RuntimeError)

  def testCheckGt(self):
    check.Gt(2, 1, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Gt(1, 1, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Gt(-1, 1, 'baz', RuntimeError)

  def testCheckLe(self):
    check.Le(1, 2, 'foo')
    check.Le(1, 1, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Le(1, 0, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Le(1, -1, 'baz', RuntimeError)

  def testCheckGe(self):
    check.Ge(2, 1, 'foo')
    check.Ge(1, 1, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Ge(0, 1, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Ge(-1, 1, 'baz', RuntimeError)

  def testCheckIs(self):
    check.Is(1, 1, 'foo')
    check.Is(None, None, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Is(1, None, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Is(1, -1, 'baz', RuntimeError)

  def testCheckIsNot(self):
    check.IsNot(1, 2, 'foo')
    check.IsNot(1, None, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsNot(None, None, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.IsNot(1, 1, 'baz', RuntimeError)

  def testCheckIsNone(self):
    check.IsNone(None, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsNone(1, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.IsNone([], 'baz', RuntimeError)

  def testCheckNotNone(self):
    check.NotNone(1, 'foo')
    check.NotNone([], 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.NotNone(None, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.NotNone(None, 'baz', RuntimeError)

  def testCheckIsTrue(self):
    check.IsTrue(1 == 1.0, 'foo')
    check.IsTrue(True, 'foo')
    check.IsTrue([0], 'foo')
    check.IsTrue({'x': 1}, 'foo')
    check.IsTrue(not 0, 'foo')
    check.IsTrue(not None, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsTrue(False, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsTrue(None, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsTrue(0, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsTrue([], 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsTrue({}, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.IsTrue('', 'baz', RuntimeError)

  def testCheckIsFalse(self):
    check.IsFalse(1 == 2, 'foo')
    check.IsFalse(False, 'foo')
    check.IsFalse([], 'foo')
    check.IsFalse({}, 'foo')
    check.IsFalse(0, 'foo')
    check.IsFalse(None, 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsFalse(True, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsFalse(not None, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsFalse(1, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsFalse([0], 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.IsFalse({'x': 1}, 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.IsFalse(' ', 'baz', RuntimeError)

  def testCheckIn(self):
    check.In('a', ('a', 'b', 'c'), 'foo')
    check.In('b', {'a': 1, 'b': 2}, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.In('d', ('a', 'b', 'c'), 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.In('c', {'a': 1, 'b': 2}, 'baz', RuntimeError)

  def testCheckNotIn(self):
    check.NotIn('d', ('a', 'b', 'c'), 'foo')
    check.NotIn('c', {'a': 1, 'b': 2}, 'bar')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.NotIn('a', ('a', 'b', 'c'), 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.NotIn('b', {'a': 1, 'b': 2}, 'baz', RuntimeError)

  def testCheckAll(self):
    check.All([], 'foo')  # empty OK
    check.All([True, 1, [1], 'hello'], 'foo')
    check.All([[[[1]]]], 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.All([None, [1], True], 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.All([True, False, True], 'baz', RuntimeError)

  def testCheckAny(self):
    check.Any([True, False, [], 'hello'], 'foo')
    check.Any([[], '', False, None, 0, 1], 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Any([None, 0, False], 'bar')
    with self.assertRaisesRegexp(ValueError, 'empty'):
      check.Any([], 'empty')  # empty not OK
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Any([0, 0.0, None], 'baz', RuntimeError)

  def testCheckSame(self):
    check.Same([], 'foo')  # empty OK
    check.Same([1, 1, 1.0, 1.0, 1], 'foo')
    check.Same(['hello', 'hello'], 'foo')
    with self.assertRaisesRegexp(ValueError, 'bar'):
      check.Same(['hello', 'world'], 'bar')
    with self.assertRaisesRegexp(RuntimeError, 'baz'):
      check.Same([1, 1.1], 'baz', RuntimeError)


if __name__ == '__main__':
  googletest.main()
