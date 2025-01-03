"""Repository rule for Python autoconfiguration.

`python_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: location of python binary.
"""

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl
    repository_ctx.template(
        out,
        Label("//third_party/py:%s.tpl" % tpl),
        substitutions,
    )

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))

def _get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin
    python_bin_path = repository_ctx.which("python")
    if python_bin_path != None:
        return str(python_bin_path)
    _fail("Cannot find python in PATH, please make sure " +
          "python is installed and add its directory in PATH, or --define " +
          "%s='/something/else'.\nPATH=%s" % (
              _PYTHON_BIN_PATH,
              repository_ctx.os.environ.get("PATH", ""),
          ))

def _create_local_python_repository(repository_ctx):
    """Creates the repository containing files set up to build with Python."""
    python_bin = _get_python_bin(repository_ctx)
    _tpl(repository_ctx, "BUILD", {
        "%{PYTHON_BIN_PATH}": python_bin,
    })

def _python_autoconf_impl(repository_ctx):
    """Implementation of the python_autoconf repository rule."""
    _create_local_python_repository(repository_ctx)

python_configure = repository_rule(
    implementation = _python_autoconf_impl,
    environ = [
        _PYTHON_BIN_PATH,
    ],
)
"""Detects and configures the local Python toolchain.

Add the following to your WORKSPACE FILE:

```python
load("//third_party/py:python_configure.bzl", "python_configure")

python_configure(name = "local_config_py_toolchain")

register_toolchains("@local_config_py_toolchain//:py_toolchain")
```

Args:
  name: A unique name for this workspace rule.
"""
