package(default_visibility = ["//skip_thoughts:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "special_words",
    srcs = ["special_words.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_binary(
    name = "preprocess_dataset",
    srcs = [
        "preprocess_dataset.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":special_words",
    ],
)
