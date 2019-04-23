# Description:
#   Build rules for Detect-to-Retrieve open source scripts.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

pytype_strict_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_binary(
    name = "extract_query_features",
    srcs = ["extract_query_features.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset",
        "//net/proto2/python/public",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:platform",
        "//third_party/tensorflow_models/delf/protos:delf_config_pb2",
        "//third_party/tensorflow_models/delf/python:feature_io",
        "//third_party/tensorflow_models/delf/python/examples:extract_features",
    ],
)

pytype_strict_binary(
    name = "extract_index_boxes_and_features",
    srcs = ["extract_index_boxes_and_features.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset",
        "//net/proto2/python/public",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:platform",
        "//third_party/tensorflow_models/delf/protos:delf_config_pb2",
        "//third_party/tensorflow_models/delf/python:box_io",
        "//third_party/tensorflow_models/delf/python:feature_io",
        "//third_party/tensorflow_models/delf/python/examples:extract_boxes",
        "//third_party/tensorflow_models/delf/python/examples:extract_features",
    ],
)

pytype_strict_binary(
    name = "cluster_delf_features",
    srcs = ["cluster_delf_features.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python:platform",
        "//third_party/tensorflow_models/delf/python:feature_io",
    ],
)
