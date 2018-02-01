"""Utils for raising errors in a CHECK-like fashion.

Example usage:

from syntaxnet.util import check

# If |num_foos| != 42, raises a ValueError with a message that contains the
# value of |num_foos|, '==', '42', and the user-provided message (in this case,
# 'Wrong number of foos').
check.Eq(num_foos, 42, 'Wrong number of foos')

# As above, but fires if |num_foos| >= 42 and raises RuntimeError.
check.Lt(num_foos, 42, 'Too many foos', error=RuntimeError)
"""


def Eq(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| does not equal |rhs|."""
  if lhs != rhs:
    raise error('Expected (%s) == (%s): %s' % (lhs, rhs, message))


def Ne(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| equals |rhs|."""
  if lhs == rhs:
    raise error('Expected (%s) != (%s): %s' % (lhs, rhs, message))


def Lt(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| is not less than |rhs|."""
  if lhs >= rhs:
    raise error('Expected (%s) < (%s): %s' % (lhs, rhs, message))


def Gt(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| is not greater than |rhs|."""
  if lhs <= rhs:
    raise error('Expected (%s) > (%s): %s' % (lhs, rhs, message))


def Le(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| is not less than or equal to |rhs|."""
  if lhs > rhs:
    raise error('Expected (%s) <= (%s): %s' % (lhs, rhs, message))


def Ge(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| is not greater than or equal to |rhs|."""
  if lhs < rhs:
    raise error('Expected (%s) >= (%s): %s' % (lhs, rhs, message))


def Is(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| is not |rhs|."""
  if lhs is not rhs:
    raise error('Expected (%s) is (%s): %s' % (lhs, rhs, message))


def IsNot(lhs, rhs, message='', error=ValueError):
  """Raises an error if |lhs| is |rhs|."""
  if lhs is rhs:
    raise error('Expected (%s) is not (%s): %s' % (lhs, rhs, message))


def IsNone(value, *args, **kwargs):
  """Raises an error if |value| is not None."""
  Is(value, None, *args, **kwargs)


def NotNone(value, *args, **kwargs):
  """Raises an error if |value| is None."""
  IsNot(value, None, *args, **kwargs)


def IsTrue(value, message='', error=ValueError):
  """Raises an error if |value| is convertible to false."""
  if not value:
    raise error('Expected (%s) to be True: %s' % (value, message))


def IsFalse(value, message='', error=ValueError):
  """Raises an error if |value| is convertible to true."""
  if value:
    raise error('Expected (%s) to be False: %s' % (value, message))


def In(key, container, message='', error=ValueError):
  """Raises an error if |key| is not in |container|."""
  if key not in container:
    raise error('Expected (%s) is in (%s): %s' % (key, container, message))


def NotIn(key, container, message='', error=ValueError):
  """Raises an error if |key| is in |container|."""
  if key in container:
    raise error('Expected (%s) is not in (%s): %s' % (key, container, message))


def All(values, message='', error=ValueError):
  """Raises an error if any of the |values| is false."""
  if not all(values):
    raise error('Expected all of %s to be true: %s' % (values, message))


def Any(values, message='', error=ValueError):
  """Raises an error if there is not one true element in |values|."""
  if not any(values):
    raise error('Expected one of %s to be true: %s' % (values, message))


def Same(values, message='', error=ValueError):
  """Raises an error if the list of |values| are not all equal."""
  if not all([value == values[0] for value in values]):
    raise error('Expected %s to equal each other: %s' % (values, message))
