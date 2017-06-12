# Skeletons for neural nets package

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "config",
    srcs = [
         "config.py"
    ],
)

py_library(
    name = "skeleton_py_pb2",
    srcs = [ "skeleton_pb2.py" ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_library(
    name = "activations",
    srcs = ["activations.py"],
    deps = [
        ":utils",
        ":skeleton_py_pb2",
    ],
)

py_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.py"],
    deps = [
        ":activations",
    ],
)

py_library(
    name = "skeleton",
    srcs = ["skeleton.py"],
    deps = [
        ":activations",
        ":skeleton_py_pb2",
    ],
)

py_test(
    name = "skeleton_test",
    size = "small",
    srcs = ["skeleton_test.py"],
    data = [
        "test_skeleton.txt",
    ],
    deps = [
        ":skeleton",
    ],
)

py_library(
    name = "neural_net",
    srcs = ["neural_net.py"],
    deps = [
        ":config",
        ":skeleton",
        ":utils",
    ],
)

py_library(
    name = "nn_trainer",
    srcs = ["nn_trainer.py"],
    deps = [
        ":neural_net",
    ],
)

py_library(
    name = "random_features",
    srcs = ["random_features.py"],
    deps = [
        ":config",
        ":skeleton",
    ],
)

py_test(
    name = "random_features_test",
    size = "small",
    srcs = ["random_features_test.py"],
    data = [
        "test_skeleton.txt",
    ],
    deps = [
        ":random_features",
    ],
)

py_library(
    name = "rf_trainer",
    srcs = ["rf_trainer.py"],
    deps = [
        ":config",
        ":random_features",
        ":skeleton",
        ":utils",
    ],
)

py_library(
    name = "kernel",
    srcs = ["kernel.py"],
    deps = [
        ":skeleton",
        ":utils",
    ],
)

py_test(
    name = "kernel_test",
    srcs = ["kernel_test.py"],
    data = [
        "simple_skeleton.pb.txt",
    ],
    deps = [
        ":kernel",
    ],
)

py_library(
    name = "concentration",
    srcs = ["concentration.py"],
    deps = [
        ":kernel",
        ":neural_net",
        ":random_features",
    ],
)
