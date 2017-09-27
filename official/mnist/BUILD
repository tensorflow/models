# Description:
# Benchmarking a model

load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_test(
    name = "mnist_test",
    srcs = ["mnist_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_logged_benchmark(
    name = "mnist_logged_benchmark",
    target = "//tensorflow/models/official/mnist:mnist_test",
)
