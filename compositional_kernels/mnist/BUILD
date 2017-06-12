# MNIST files for working with skeletons

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "mnist_data_provider",
    srcs = ["mnist_data_provider.py"],
    deps = [
        "//base:config",
    ],
)

py_test(
    name = "mnist_data_provider_test",
    size = "large",
    srcs = ["mnist_data_provider_test.py"],
    data = [
        "datasets/t10k-images-idx3-ubyte.gz",
        "datasets/t10k-labels-idx1-ubyte.gz",
        "datasets/train-images-idx3-ubyte.gz",
        "datasets/train-labels-idx1-ubyte.gz",
    ],
    deps = [
        ":mnist_data_provider",
    ],
)

py_binary(
    name = "nn_trainer",
    srcs = ["nn_trainer.py"],
    deps = [
        ":mnist_data_provider",
        "//base:nn_trainer",
    ],
)

py_binary(
    name = "rf_trainer",
    srcs = ["rf_trainer.py"],
    deps = [
        ":mnist_data_provider",
        "//base:rf_trainer",
    ],
)

py_binary(
    name = "concentration",
    srcs = ["concentration.py"],
    deps = [
        ":mnist_data_provider",
        "//base:concentration",
    ],
)
