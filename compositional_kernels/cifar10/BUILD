# CIFAR10 files for working with skeletons

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "cifar10_data_provider",
    srcs = ["cifar10_data_provider.py"],
    deps = [
        "//base:config",
    ],
)

py_test(
    name = "cifar10_data_provider_test",
    size = "large",
    srcs = ["cifar10_data_provider_test.py"],
    data = [
        "datasets/cifar-10-batches-bin/data_batch_1.bin",
        "datasets/cifar-10-batches-bin/data_batch_2.bin",
        "datasets/cifar-10-batches-bin/data_batch_3.bin",
        "datasets/cifar-10-batches-bin/data_batch_4.bin",
        "datasets/cifar-10-batches-bin/data_batch_5.bin",
        "datasets/cifar-10-batches-bin/test_batch.bin",
    ],
    deps = [
        ":cifar10_data_provider",
    ],
)

py_binary(
    name = "nn_trainer",
    srcs = ["nn_trainer.py"],
    deps = [
        ":cifar10_data_provider",
        "//base:nn_trainer",
    ],
)

py_binary(
    name = "rf_trainer",
    srcs = ["rf_trainer.py"],
    deps = [
        ":cifar10_data_provider",
        "//base:rf_trainer",
    ],
)

py_binary(
    name = "concentration",
    srcs = ["concentration.py"],
    deps = [
        ":cifar10_data_provider",
        "//base:concentration",
    ],
)
