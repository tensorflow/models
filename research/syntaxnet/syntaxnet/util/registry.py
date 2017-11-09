"""A component registry, similar to nlp_saft::RegisteredClass<>.

Like nlp_saft::RegisteredClass<>, one does not need to explicitly import the
module containing each subclass.  It is sufficient to add subclasses as build
dependencies.

Unlike nlp_saft::RegisteredClass<>, which allows subclasses to be registered
under arbitrary names, subclasses must be looked up based on their type name.
This restriction allows the registry to dynamically import the module containing
the desired subclass.

Example usage:

# In basepackage/base.py...
@registry.RegisteredClass
class MyBase:
  def my_method(self):
    pass

# In implpackage/impl.py...
class MyImpl(MyBase):
  def my_method(self):
    ...

# In userpackage/user.py...
try
  impl = MyBase.Create("implpackage.impl.MyImpl")
except ValueError as error:
  ...

Note that there is no registration statement in impl.py.  For convenience, if
the base class and subclass share a package prefix, the shared portion of the
package path may be omitted in the call to Create().  For example, if the base
class is 'foo.bar.Base' and the subclass is 'foo.bar.baz.Impl', then these are
all equivalent:

  Base.Create('foo.bar.baz.Impl')
  Base.Create('bar.baz.Impl')
  Base.Create('baz.Impl')

Name resolution happens in inside-out fashion, so if there is also a subclass
'foo.baz.Impl', then

  Base.Create('baz.Impl')      # returns foo.bar.baz.Impl
  Base.Create('bar.baz.Impl')  # returns foo.bar.baz.Impl
  Base.Create('foo.baz.Impl')  # returns foo.baz.Impl

NB: Care is required when moving code, because config files may refer to the
classes being moved by their type name, which may include the package path.  To
preserve existing names, leave a stub in the original location that imports the
class from its new location.  For example,

# Before move, in oldpackage/old.py...
class Foo(Base):
  ...

# After move, in newpackage/new.py...
class Bar(Base):
  ...

# After move, in oldpackage/old.py...
from newpackage import new
Foo = new.Bar
"""

import inspect
import sys

from tensorflow.python.platform import tf_logging as logging


def _GetClass(name):
  """Looks up a class by name.

  Args:
    name: The fully-qualified type name of the class to return.

  Returns:
    The class associated with the |name|, or None on error.
  """
  elements = name.split('.')

  # Need at least "module.Class".
  if len(elements) < 2:
    logging.debug('Malformed type: "%s"', name)
    return None
  module_path = '.'.join(elements[:-1])
  class_name = elements[-1]

  # Import the module.
  try:
    __import__(module_path)
  except ImportError as e:
    logging.debug('Unable to find module "%s": "%s"', module_path, e)
    return None
  module = sys.modules[module_path]

  # Look up the class.
  if not hasattr(module, class_name):
    logging.debug('Name "%s" not found in module: "%s"', class_name,
                  module_path)
    return None
  class_obj = getattr(module, class_name)

  # Check that it is actually a class.
  if not inspect.isclass(class_obj):
    logging.debug('Name does not refer to a class: "%s"', name)
    return None
  return class_obj


def _Create(baseclass, subclass_name, *args, **kwargs):
  """Creates an instance of a named subclass.

  Args:
    baseclass: The expected base class.
    subclass_name: The fully-qualified type name of the subclass to create.
    *args: Passed to the subclass constructor.
    **kwargs: Passed to the subclass constructor.

  Returns:
    An instance of the named subclass, or None on error.
  """
  subclass = _GetClass(subclass_name)
  if subclass is None:
    return None  # _GetClass() already logged an error
  if not issubclass(subclass, baseclass):
    logging.debug('Class "%s" is not a subclass of "%s"', subclass_name,
                  baseclass.__name__)
    return None
  return subclass(*args, **kwargs)


def _ResolveAndCreate(baseclass, path, subclass_name, *args, **kwargs):
  """Resolves the name of a subclass and creates an instance of it.

  The subclass is resolved with respect to a package path in an inside-out
  manner.  For example, if |path| is 'google3.foo.bar' and |subclass_name| is
  'baz.ClassName', then attempts are made to create instances of the following
  fully-qualified class names:

    'google3.foo.bar.baz.ClassName'
    'google3.foo.baz.ClassName'
    'google3.baz.ClassName'
    'baz.ClassName'

  An instance corresponding to the first successful attempt is returned.

  Args:
    baseclass: The expected base class.
    path: The path to use to resolve the subclass.
    subclass_name: The name of the subclass to create.
    *args: Passed to the subclass constructor.
    **kwargs: Passed to the subclass constructor.

  Returns:
    An instance of the named subclass corresponding to the inner-most successful
    name resolution, or None if the name could not be resolved.

  Raises:
    ValueError: If the subclass cannot be resolved and created.
  """
  elements = path.split('.')
  while True:
    resolved_subclass_name = '.'.join(elements + [subclass_name])
    subclass = _Create(baseclass, resolved_subclass_name, *args, **kwargs)
    if subclass: return subclass  # success
    if not elements: break  # no more paths to try
    elements.pop()  # try resolving against the next-outer path
  raise ValueError(
      'Failed to create subclass "%s" of base class %s using path %s' %
      (subclass_name, baseclass.__name__, path))


def RegisteredClass(baseclass):
  """Decorates the |baseclass| with a static Create() method."""
  assert not hasattr(baseclass, 'Create')

  def Create(subclass_name, *args, **kwargs):
    """A wrapper around _Create() that curries the |baseclass|."""
    path = inspect.getmodule(baseclass).__name__
    return _ResolveAndCreate(baseclass, path, subclass_name, *args, **kwargs)

  baseclass.Create = staticmethod(Create)
  return baseclass
