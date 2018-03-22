package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//astronet/astro_model:configurations",
    ],
)

py_library(
    name = "astro_fc_model",
    srcs = [
        "astro_fc_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//astronet/astro_model"],
)

py_test(
    name = "astro_fc_model_test",
    size = "small",
    srcs = [
        "astro_fc_model_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":astro_fc_model",
        ":configurations",
        "//astronet/ops:input_ops",
        "//astronet/ops:testing",
        "//astronet/util:configdict",
    ],
)
