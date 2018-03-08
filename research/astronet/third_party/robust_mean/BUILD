package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD

exports_files(["LICENSE"])

py_library(
    name = "robust_mean",
    srcs = ["robust_mean.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "robust_mean_test",
    size = "small",
    srcs = [
        "robust_mean_test.py",
        "test_data/random_normal.py",
    ],
    srcs_version = "PY2AND3",
    deps = [":robust_mean"],
)
