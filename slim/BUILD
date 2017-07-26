# Description:
#   Contains files for loading, training and evaluating TF-Slim-based models.

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(name = "internal")

py_library(
    name = "dataset_utils",
    srcs = ["datasets/dataset_utils.py"],
)

py_library(
    name = "download_and_convert_cifar10",
    srcs = ["datasets/download_and_convert_cifar10.py"],
    deps = [":dataset_utils"],
)

py_library(
    name = "download_and_convert_flowers",
    srcs = ["datasets/download_and_convert_flowers.py"],
    deps = [":dataset_utils"],
)

py_library(
    name = "download_and_convert_mnist",
    srcs = ["datasets/download_and_convert_mnist.py"],
    deps = [":dataset_utils"],
)

py_binary(
    name = "download_and_convert_data",
    srcs = ["download_and_convert_data.py"],
    deps = [
        ":download_and_convert_cifar10",
        ":download_and_convert_flowers",
        ":download_and_convert_mnist",
    ],
)

py_binary(
    name = "cifar10",
    srcs = ["datasets/cifar10.py"],
    deps = [":dataset_utils"],
)

py_binary(
    name = "flowers",
    srcs = ["datasets/flowers.py"],
    deps = [":dataset_utils"],
)

py_binary(
    name = "imagenet",
    srcs = ["datasets/imagenet.py"],
    deps = [":dataset_utils"],
)

py_binary(
    name = "mnist",
    srcs = ["datasets/mnist.py"],
    deps = [":dataset_utils"],
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

py_library(
    name = "model_deploy",
    srcs = ["deployment/model_deploy.py"],
)

py_test(
    name = "model_deploy_test",
    srcs = ["deployment/model_deploy_test.py"],
    srcs_version = "PY2AND3",
    deps = [":model_deploy"],
)

py_library(
    name = "cifarnet_preprocessing",
    srcs = ["preprocessing/cifarnet_preprocessing.py"],
)

py_library(
    name = "inception_preprocessing",
    srcs = ["preprocessing/inception_preprocessing.py"],
)

py_library(
    name = "lenet_preprocessing",
    srcs = ["preprocessing/lenet_preprocessing.py"],
)

py_library(
    name = "vgg_preprocessing",
    srcs = ["preprocessing/vgg_preprocessing.py"],
)

py_library(
    name = "preprocessing_factory",
    srcs = ["preprocessing/preprocessing_factory.py"],
    deps = [
        ":cifarnet_preprocessing",
        ":inception_preprocessing",
        ":lenet_preprocessing",
        ":vgg_preprocessing",
    ],
)

# Typical networks definitions.

py_library(
    name = "nets",
    deps = [
        ":alexnet",
        ":cifarnet",
        ":inception",
        ":lenet",
        ":mobilenet_v1",
        ":overfeat",
        ":resnet_v1",
        ":resnet_v2",
        ":vgg",
    ],
)

py_library(
    name = "alexnet",
    srcs = ["nets/alexnet.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "alexnet_test",
    size = "medium",
    srcs = ["nets/alexnet_test.py"],
    srcs_version = "PY2AND3",
    deps = [":alexnet"],
)

py_library(
    name = "cifarnet",
    srcs = ["nets/cifarnet.py"],
)

py_library(
    name = "inception",
    srcs = ["nets/inception.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_resnet_v2",
        ":inception_v1",
        ":inception_v2",
        ":inception_v3",
        ":inception_v4",
    ],
)

py_library(
    name = "inception_utils",
    srcs = ["nets/inception_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "inception_v1",
    srcs = ["nets/inception_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
    ],
)

py_library(
    name = "inception_v2",
    srcs = ["nets/inception_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
    ],
)

py_library(
    name = "inception_v3",
    srcs = ["nets/inception_v3.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
    ],
)

py_library(
    name = "inception_v4",
    srcs = ["nets/inception_v4.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
    ],
)

py_library(
    name = "inception_resnet_v2",
    srcs = ["nets/inception_resnet_v2.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "inception_v1_test",
    size = "large",
    srcs = ["nets/inception_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [":inception"],
)

py_test(
    name = "inception_v2_test",
    size = "large",
    srcs = ["nets/inception_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [":inception"],
)

py_test(
    name = "inception_v3_test",
    size = "large",
    srcs = ["nets/inception_v3_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [":inception"],
)

py_test(
    name = "inception_v4_test",
    size = "large",
    srcs = ["nets/inception_v4_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [":inception"],
)

py_test(
    name = "inception_resnet_v2_test",
    size = "large",
    srcs = ["nets/inception_resnet_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [":inception"],
)

py_library(
    name = "lenet",
    srcs = ["nets/lenet.py"],
)

py_library(
    name = "mobilenet_v1",
    srcs = ["nets/mobilenet_v1.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "mobilenet_v1_test",
    size = "large",
    srcs = ["nets/mobilenet_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_v1",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["nets/overfeat.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "overfeat_test",
    size = "medium",
    srcs = ["nets/overfeat_test.py"],
    srcs_version = "PY2AND3",
    deps = [":overfeat"],
)

py_library(
    name = "resnet_utils",
    srcs = ["nets/resnet_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "resnet_v1",
    srcs = ["nets/resnet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
    ],
)

py_test(
    name = "resnet_v1_test",
    size = "medium",
    srcs = ["nets/resnet_v1_test.py"],
    srcs_version = "PY2AND3",
    deps = [":resnet_v1"],
)

py_library(
    name = "resnet_v2",
    srcs = ["nets/resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
    ],
)

py_test(
    name = "resnet_v2_test",
    size = "medium",
    srcs = ["nets/resnet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [":resnet_v2"],
)

py_library(
    name = "vgg",
    srcs = ["nets/vgg.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "vgg_test",
    size = "medium",
    srcs = ["nets/vgg_test.py"],
    srcs_version = "PY2AND3",
    deps = [":vgg"],
)

py_library(
    name = "nets_factory",
    srcs = ["nets/nets_factory.py"],
    deps = [":nets"],
)

py_test(
    name = "nets_factory_test",
    size = "medium",
    srcs = ["nets/nets_factory_test.py"],
    srcs_version = "PY2AND3",
    deps = [":nets_factory"],
)

py_binary(
    name = "train_image_classifier",
    srcs = ["train_image_classifier.py"],
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":nets_factory",
        ":preprocessing_factory",
    ],
)

py_binary(
    name = "eval_image_classifier",
    srcs = ["eval_image_classifier.py"],
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":nets_factory",
        ":preprocessing_factory",
    ],
)

py_binary(
    name = "export_inference_graph",
    srcs = ["export_inference_graph.py"],
    deps = [
        ":dataset_factory",
        ":nets_factory",
    ],
)

py_test(
    name = "export_inference_graph_test",
    size = "medium",
    srcs = ["export_inference_graph_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
    ],
    deps = [
        ":export_inference_graph",
        ":nets_factory",
    ],
)
