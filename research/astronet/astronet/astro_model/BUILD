package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "astro_model",
    srcs = [
        "astro_model.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "astro_model_test",
    size = "small",
    srcs = [
        "astro_model_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":astro_model",
        ":configurations",
        "//astronet/ops:input_ops",
        "//astronet/ops:testing",
        "//astronet/util:configdict",
    ],
)
