licenses(["notice"])

package(
    default_visibility = [
        "//:__subpackages__",
    ],
)

py_library(
    name = "common_layer",
    srcs = ["common_layer.py"],
    srcs_version = "PY3",
)

py_library(
    name = "prado_model",
    srcs = ["prado_model.py"],
    srcs_version = "PY3",
    deps = [
        ":common_layer",
        "//tf_ops:sequence_string_projection_op_py",
    ],
)

py_library(
    name = "metric_functions",
    srcs = ["metric_functions.py"],
    srcs_version = "PY3",
)

py_library(
    name = "input_fn_reader",
    srcs = ["input_fn_reader.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "runner",
    srcs = ["runner.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_fn_reader",
        ":metric_functions",
        ":prado_model",
    ],
)
