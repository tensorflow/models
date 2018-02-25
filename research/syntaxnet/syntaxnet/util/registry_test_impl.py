"""A dummy implementation for use in RegistryTest."""

from syntaxnet.util import registry_test_base


class Impl(registry_test_base.Base):
  """Dummy implementation."""

  def __init__(self, value):
    """Creates an implementation with a custom string."""
    self.value = value

  def Get(self):
    """Returns the current value."""
    return self.value


# An alias for another class.
Alias = Impl  # NOLINT


class NonSubclass(object):
  """A class that is not a subclass of registry_test_base.Base."""
  pass


# A dummy variable, to exercise type checking.
variable = 1


def Function():
  """A dummy function, to exercise type checking."""
  pass
