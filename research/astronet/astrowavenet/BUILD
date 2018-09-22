"""A TensorFlow model for generative modeling of light curves."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "astrowavenet",
    srcs = [
        "astrowavenet.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "astrowavenet_test",
    size = "small",
    srcs = [
        "astrowavenet_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":astrowavenet",
        ":configurations",
        "//astronet/util:configdict",
    ],
)
