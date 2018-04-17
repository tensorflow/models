package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "input_ops",
    srcs = ["input_ops.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "input_ops_test",
    size = "small",
    srcs = ["input_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":input_ops",
        "//astronet/util:configdict",
    ],
)

py_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "dataset_ops_test",
    size = "small",
    srcs = ["dataset_ops_test.py"],
    data = ["test_data/test_dataset.tfrecord"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_ops",
        "//astronet/util:configdict",
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "testing",
    srcs = ["testing.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.py"],
    srcs_version = "PY2AND3",
    deps = [":metrics"],
)
