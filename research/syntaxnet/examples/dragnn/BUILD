py_binary(
    name = "tutorial_1",
    srcs = ["tutorial_1.py"],
    data = [":data"],
    deps = [":tutorial-deps"],
)

py_binary(
    name = "tutorial_2",
    srcs = ["tutorial_2.py"],
    data = [":data"],
    deps = [":tutorial-deps"],
)

py_library(
    name = "tutorial-deps",
    deps = [
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//dragnn/protos:spec_pb2_py",
        "//dragnn/python:graph_builder",
        "//dragnn/python:lexicon",
        "//dragnn/python:load_dragnn_cc_impl_py",
        "//dragnn/python:spec_builder",
        "//dragnn/python:visualization",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:parser_ops",
        "//syntaxnet:sentence_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

filegroup(
    name = "data",
    data = glob(["tutorial_data/*"]),
    visibility = ["//visibility:public"],
)

sh_test(
    name = "test_run_all_tutorials",
    size = "medium",
    srcs = ["test_run_all_tutorials.sh"],
    args = [
        "$(location :tutorial_1)",
        "$(location :tutorial_2)",
    ],
    data = [
        ":tutorial_1",
        ":tutorial_2",
    ],
)
