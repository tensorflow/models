package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "configdict",
    srcs = ["configdict.py"],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_test(
    name = "configdict_test",
    size = "small",
    srcs = ["configdict_test.py"],
    srcs_version = "PY2AND3",
    deps = [":configdict"],
)

py_library(
    name = "config_util",
    srcs = ["config_util.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "config_util_test",
    size = "small",
    srcs = ["config_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [":config_util"],
)

py_library(
    name = "estimator_util",
    srcs = ["estimator_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//astronet/ops:dataset_ops",
        "//astronet/ops:metrics",
        "//astronet/ops:training",
    ],
)
