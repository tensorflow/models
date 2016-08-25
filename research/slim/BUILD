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

py_library(
    name = "dataset_utils",
    srcs = ["datasets/dataset_utils.py"],
    deps = [
        "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_cifar10",
    srcs = ["datasets/download_and_convert_cifar10.py"],
    deps = [
        ":dataset_utils",
        "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_flowers",
    srcs = ["datasets/download_and_convert_flowers.py"],
    deps = [
        ":dataset_utils",
        "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_mnist",
    srcs = ["datasets/download_and_convert_mnist.py"],
    deps = [
        ":dataset_utils",
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

py_binary(
    name = "flowers",
    srcs = ["datasets/flowers.py"],
    deps = [
        "//tensorflow",
    ],
)

py_binary(
    name = "imagenet",
    srcs = ["datasets/imagenet.py"],
    deps = [
        "//tensorflow",
    ],
)

py_binary(
    name = "mnist",
    srcs = ["datasets/mnist.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "dataset_factory",
    srcs = ["datasets/dataset_factory.py"],
    deps = [
        ":cifar10",
        ":flowers",
        ":imagenet",
        ":mnist",
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":model_factory",
        "//tensorflow",
    ],
)

py_library(
    name = "inception_preprocessing",
    srcs = ["models/inception_preprocessing.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "model_deploy",
    srcs = ["models/model_deploy.py"],
    deps = [
        "//tensorflow",
    ],
)

py_test(
    name = "model_deploy_test",
    srcs = ["models/model_deploy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_deploy",
        "//tensorflow",
    ],
)

py_library(
    name = "inception_preprocessing",
    srcs = ["models/inception_preprocessing.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "resnet_preprocessing",
    srcs = ["models/resnet_preprocessing.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "model_factory",
    srcs = ["models/model_factory.py"],
    deps = [
        ":inception_preprocessing",
        ":resnet_preprocessing",
        "//tensorflow",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":model_factory",
        "//tensorflow",
    ],
)
