package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//differential_privacy/...",
    ],
)

py_library(
    name = "aggregation",
    srcs = [
        "aggregation.py",
    ],
    deps = [
    ],
)

py_library(
    name = "deep_cnn",
    srcs = [
        "deep_cnn.py",
    ],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "input",
    srcs = [
        "input.py",
    ],
    deps = [
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "metrics.py",
    ],
    deps = [
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    deps = [
    ],
)

py_binary(
    name = "train_student",
    srcs = [
        "train_student.py",
    ],
    deps = [
        ":aggregation",
        ":deep_cnn",
        ":input",
        ":metrics",
    ],
)

py_binary(
    name = "train_teachers",
    srcs = [
        "train_teachers.py",
        ":deep_cnn",
        ":input",
        ":metrics",
    ],
    deps = [
    ],
)

py_library(
    name = "analysis",
    srcs = [
        "analysis.py",
    ],
    deps = [
        "//differential_privacy/multiple_teachers:input",
    ],
)
