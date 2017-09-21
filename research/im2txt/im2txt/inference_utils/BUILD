package(default_visibility = ["//im2txt:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "inference_wrapper_base",
    srcs = ["inference_wrapper_base.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "vocabulary",
    srcs = ["vocabulary.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "caption_generator",
    srcs = ["caption_generator.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "caption_generator_test",
    srcs = ["caption_generator_test.py"],
    deps = [
        ":caption_generator",
    ],
)
