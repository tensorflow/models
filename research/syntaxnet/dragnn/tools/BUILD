package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob(["testdata/**"]),
)

py_binary(
    name = "conll_checkpoint_converter",
    srcs = ["conll_checkpoint_converter.py"],
    deps = [
        "//dragnn/protos:spec_py_pb2",
        "//dragnn/python:dragnn_model_saver_lib",
        "//dragnn/python:spec_builder",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_binary(
    name = "evaluator",
    srcs = ["evaluator.py"],
    tags = [
        "notap",
        "optonly",
    ],
    deps = [
        ":components",
        "//dragnn/python:evaluation",
        "//dragnn/python:spec_builder",
    ],
)

py_binary(
    name = "legacy_parse_to_conll",
    srcs = ["legacy_parse_to_conll.py"],
    tags = [
        "notap",
        "optonly",
    ],
    deps = [
        ":components",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:spec_builder",
    ],
)

py_binary(
    name = "segmenter-evaluator",
    srcs = ["segmenter-evaluator.py"],
    tags = [
        "notap",
        "optonly",
    ],
    deps = [
        ":components",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:spec_builder",
    ],
)

py_binary(
    name = "parse_to_conll",
    srcs = ["parse_to_conll.py"],
    tags = [
        "notap",
        "optonly",
    ],
    deps = [
        ":components",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:spec_builder",
    ],
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        ":components",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:lexicon",
        "//dragnn/python:spec_builder",
        "//dragnn/python:trainer_lib",
        "//syntaxnet:task_spec_py_pb2",
    ],
)

py_binary(
    name = "parser_trainer",
    srcs = ["parser_trainer.py"],
    deps = [
        ":components",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:lexicon",
        "//dragnn/python:spec_builder",
        "//dragnn/python:trainer_lib",
        "//syntaxnet:task_spec_py_pb2",
    ],
)

py_binary(
    name = "segmenter_trainer",
    srcs = ["segmenter_trainer.py"],
    deps = [
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//dragnn/protos:spec_py_pb2",
        "//dragnn/python:evaluation",
        "//dragnn/python:graph_builder",
        "//dragnn/python:sentence_io",
        "//dragnn/python:spec_builder",
        "//dragnn/python:trainer_lib",
        "//syntaxnet:parser_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_binary(
    name = "model_trainer",
    srcs = ["model_trainer.py"],
    deps = [
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//dragnn/protos:spec_py_pb2",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:graph_builder",
        "//dragnn/python:sentence_io",
        "//dragnn/python:spec_builder",
        "//dragnn/python:trainer_lib",
        "//syntaxnet:parser_ops",
        "//syntaxnet:sentence_py_pb2",
        "//syntaxnet:task_spec_py_pb2",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

sh_test(
    name = "model_trainer_test",
    size = "medium",
    srcs = ["model_trainer_test.sh"],
    data = [
        ":model_trainer",
        ":testdata",
    ],
    deps = [
    ],
)

# This is meant to be run inside the Docker image. In the OSS directory, run,
#
#     ./build_devel.sh bazel run //dragnn/python:oss_notebook_launcher
py_binary(
    name = "oss_notebook_launcher",
    srcs = ["oss_notebook_launcher.py"],
    args = [
        "notebook",
        "--notebook-dir=/opt/tensorflow/syntaxnet/examples",
        "--no-browser",
    ],
    tags = [
        "optonly",
    ],
    deps = [
        ":all_in_one_components",
    ],
)

py_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.py"],
    data = ["oss_setup.py"],
    deps = [
        ":all_in_one_components",
    ],
)

# Shared/common components for all tools.
py_library(
    name = "components",
    deps = [
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//dragnn/protos:spec_py_pb2",
        "//dragnn/python:graph_builder",
        "//dragnn/python:sentence_io",
        "//syntaxnet:parser_ops",
        "//syntaxnet:sentence_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

# This should include just about everything, suitable for Jupyter notebooks or
# building a pip package.
py_library(
    name = "all_in_one_components",
    deps = [
        ":components",
        "//dragnn/python:components",
        "//dragnn/python:dragnn_ops",
        "//dragnn/python:evaluation",
        "//dragnn/python:lexicon",
        "//dragnn/python:render_parse_tree_graphviz",
        "//dragnn/python:render_spec_with_graphviz",
        "//dragnn/python:spec_builder",
        "//dragnn/python:trainer_lib",
        "//dragnn/python:visualization",
        "//syntaxnet:task_spec_py_pb2",
    ],
)
