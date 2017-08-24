# Description:
# Python support for TensorFlow.

package(default_visibility = ["//third_party/tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":reader",
    ],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/tensorflow"],
)

py_test(
    name = "reader_test",
    size = "small",
    srcs = ["reader_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":reader",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/py/tensorflow"],
)

py_binary(
    name = "ptb_word_lm",
    srcs = [
        "ptb_word_lm.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":reader",
        ":util",
        "//third_party/py/tensorflow",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//third_party/tensorflow:__subpackages__"],
)
