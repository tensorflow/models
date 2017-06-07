# Description:
#   Contains the operations and nets for building TensorFlow-Slim models.

package(default_visibility = ["//vgg:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "scopes",
    srcs = ["scopes.py"],
)

py_test(
    name = "scopes_test",
    size = "small",
    srcs = ["scopes_test.py"],
    deps = [
        ":scopes",
    ],
)

py_library(
    name = "variables",
    srcs = ["variables.py"],
    deps = [
        ":scopes",
    ],
)

py_test(
    name = "variables_test",
    size = "small",
    srcs = ["variables_test.py"],
    deps = [
        ":variables",
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
)

py_test(
    name = "losses_test",
    size = "small",
    srcs = ["losses_test.py"],
    deps = [
        ":losses",
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":losses",
        ":scopes",
        ":variables",
    ],
)

py_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.py"],
    deps = [
        ":ops",
        ":variables",
    ],
)

py_library(
    name = "vgg",
    srcs = ["vgg_model.py"],
    deps = [
        ":ops",
        ":scopes",
    ],
)

py_library(
    name = "slim",
    srcs = ["slim.py"],
    deps = [
        ":vgg",
        ":losses",
        ":ops",
        ":scopes",
        ":variables",
    ],
)

