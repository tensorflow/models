licenses(["notice"])

package(
    default_visibility = [
        "//:__subpackages__",
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
    deps = [
        "//layers:projection_layers",
    ],
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_fn_reader",
        ":metric_functions",
        "//models:prado",
    ],
)

py_binary(
    name = "export_to_tflite",
    srcs = ["export_to_tflite.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":input_fn_reader",
        ":metric_functions",
        "//layers:base_layers",
        "//layers:projection_layers",
        "//models:prado",
        "//utils:tflite_utils",
    ],
)
