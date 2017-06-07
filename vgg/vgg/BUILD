# Description:
# Example TensorFlow models for ImageNet.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//vgg/..."],
)

py_library(
    name = "dataset",
    srcs = [
        "dataset.py",
    ],
)

py_library(
    name = "imagenet_data",
    srcs = [
        "imagenet_data.py",
    ],
    deps = [
        ":dataset",
    ],
)

py_library(
    name = "image_processing",
    srcs = [
        "image_processing.py",
    ],
)

py_library(
    name = "vgg16",
    srcs = [
        "vgg16_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        "//inception/slim",
    ],
)

py_binary(
    name = "imagenet_eval",
    srcs = [
        "imagenet_eval.py",
    ],
    deps = [
        ":imagenet_data",
        ":vgg16_eval",
    ],
)

py_library(
    name = "vgg16_eval",
    srcs = [
        "vgg16_eval.py",
    ],
    deps = [
        ":image_processing",
        ":vgg16",
    ],
)

py_binary(
    name = "imagenet_train",
    srcs = [
        "imagenet_train.py",
    ],
    deps = [
        ":imagenet_data",
        ":vgg16_train",
    ],
)

py_binary(
    name = "imagenet_distributed_train",
    srcs = [
        "imagenet_distributed_train.py",
    ],
    deps = [
        ":imagenet_data",
        ":vgg16_distributed_train",
    ],
)

py_library(
    name = "vgg16_train",
    srcs = [
        "vgg16_train.py",
    ],
    deps = [
        ":image_processing",
        ":vgg16",
    ],
)

py_library(
    name = "vgg16_distributed_train",
    srcs = [
        "vgg16_distributed_train.py",
    ],
    deps = [
        ":image_processing",
        ":vgg16",
    ],
)

py_binary(
    name = "build_image_data",
    srcs = ["data/build_image_data.py"],
)

sh_binary(
    name = "download_and_preprocess_imagenet",
    srcs = ["data/download_and_preprocess_imagenet.sh"],
    data = [
        "data/download_imagenet.sh",
        "data/imagenet_2012_validation_synset_labels.txt",
        "data/imagenet_lsvrc_2015_synsets.txt",
        "data/imagenet_metadata.txt",
        "data/preprocess_imagenet_validation_data.py",
        "data/process_bounding_boxes.py",
        ":build_imagenet_data",
    ],
)

py_binary(
    name = "build_imagenet_data",
    srcs = ["data/build_imagenet_data.py"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "**/*.py",
            "BUILD",
        ],
    ),
)

filegroup(
    name = "imagenet_metadata",
    srcs = [
        "data/imagenet_lsvrc_2015_synsets.txt",
        "data/imagenet_metadata.txt",
    ],
    visibility = ["//visibility:public"],
)
