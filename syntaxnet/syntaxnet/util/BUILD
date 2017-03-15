# Utilities.
package(default_visibility = ["//visibility:public"])

py_library(
    name = "pyregistry",
    srcs = ["registry.py"],
    deps = [
        "//syntaxnet:load_parser_ops_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "check",
    srcs = ["check.py"],
)

py_library(
    name = "pyregistry_test_base",
    testonly = 1,
    srcs = ["registry_test_base.py"],
)

py_library(
    name = "pyregistry_test_impl",
    testonly = 1,
    srcs = ["registry_test_impl.py"],
    deps = [
        ":pyregistry_test_base",
    ],
)

# Note: py_test() requires the name of the build rule and the source file to
# match, so this is called "registry_test" instead of "pyregistry_test".
py_test(
    name = "registry_test",
    srcs = ["registry_test.py"],
    deps = [
        ":pyregistry",
        ":pyregistry_test_base",
        ":pyregistry_test_impl",
        "//syntaxnet:load_parser_ops_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "check_test",
    srcs = ["check_test.py"],
    deps = [
        ":check",
        "//syntaxnet:load_parser_ops_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)
