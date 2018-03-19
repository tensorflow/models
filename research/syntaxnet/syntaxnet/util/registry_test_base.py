"""A dummy base class for use in RegistryTest."""

from syntaxnet.util import registry


@registry.RegisteredClass
class Base(object):
  """Dummy base class."""

  def Get(self):
    """Overridden in subclasses."""
    return None
