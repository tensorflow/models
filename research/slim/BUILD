# Description:
#   Contains files for loading, training and evaluating TF-Slim 2.0-based models.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//learning/brain/contrib/slim...",
        "//research/vale/tensorflow/slim/...",
    ],
)

py_binary(
    name = "create_cifar10_dataset",
    srcs = ["data/create_cifar10_dataset.py"],
    deps = [
        "//tensorflow",
    ],
)

py_binary(
    name = "cifar10",
    srcs = ["datasets/cifar10.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "dataset_factory",
    srcs = ["datasets/dataset_factory.py"],
    deps = [
        ":cifar10",
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":dataset_factory",
        ":model_factory",
        ":preprocessing",
        "//tensorflow",
    ],
)

py_library(
    name = "model_factory",
    srcs = ["models/model_factory.py"],
    deps = [
        ":preprocessing",
        "//tensorflow",
    ],
)

py_binary(
    name = "preprocessing",
    srcs = ["preprocessing.py"],
    deps = [
        "//tensorflow",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset_factory",
        ":model_factory",
        ":preprocessing",
        "//tensorflow",
    ],
)
