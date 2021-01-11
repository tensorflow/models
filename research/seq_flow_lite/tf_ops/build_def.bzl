def tf_deps():
    return [
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ]

def tf_copts():
    return ["-Wno-sign-compare"]

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.
    levels_to_root = native.package_name().count("/") + name.count("/")
    return ["-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),)]

def gen_op_wrapper_py(name, out, kernel_lib, linkopts = [], **kwargs):
    """Generates the py_library `name` with a data dep on the ops in kernel_lib.

    The resulting py_library creates file `$out`, and has a dependency on a
    symbolic library called lib{$name}_gen_op.so, which contains the kernels
    and ops and can be loaded via `tf.load_op_library`.

    Args:
      name: The name of the py_library.
      out: The name of the python file.  Use "gen_{name}_ops.py".
      kernel_lib: A cc_kernel_library target to generate for.
      **kwargs: Any args to the `cc_binary` and `py_library` internal rules.
    """
    if not out.endswith(".py"):
        fail("Argument out must end with '.py', but saw: {}".format(out))

    module_name = "lib{}_gen_op".format(name)
    version_script_file = "%s-version-script.lds" % module_name
    native.genrule(
        name = module_name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n *tensorflow*;\n *deepmind*;\n local: *;};' >$@",
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )
    native.cc_binary(
        name = "{}.so".format(module_name),
        deps = [kernel_lib] + tf_deps() + [version_script_file],
        copts = tf_copts() + [
            "-fno-strict-aliasing",  # allow a wider range of code [aliasing] to compile.
            "-fvisibility=hidden",  # avoid symbol clashes between DSOs.
        ],
        linkshared = 1,
        linkopts = linkopts + _rpath_linkopts(module_name) + [
            "-Wl,--version-script",
            "$(location %s)" % version_script_file,
        ],
        **kwargs
    )
    native.genrule(
        name = "{}_genrule".format(out),
        outs = [out],
        cmd = """
        echo 'import tensorflow as tf
_reverb_gen_op = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile(
       "lib{}_gen_op.so"))
_locals = locals()
for k in dir(_reverb_gen_op):
  _locals[k] = getattr(_reverb_gen_op, k)
del _locals' > $@""".format(name),
    )
    native.py_library(
        name = name,
        srcs = [out],
        data = [":lib{}_gen_op.so".format(name)],
        **kwargs
    )
