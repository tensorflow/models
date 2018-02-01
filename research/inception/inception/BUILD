# Description:
# Example TensorFlow models for ImageNet.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//inception/..."],
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
    name = "flowers_data",
    srcs = [
        "flowers_data.py",
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
    name = "inception",
    srcs = [
        "inception_model.py",
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
        ":inception_eval",
    ],
)

py_binary(
    name = "flowers_eval",
    srcs = [
        "flowers_eval.py",
    ],
    deps = [
        ":flowers_data",
        ":inception_eval",
    ],
)

py_library(
    name = "inception_eval",
    srcs = [
        "inception_eval.py",
    ],
    deps = [
        ":image_processing",
        ":inception",
    ],
)

py_binary(
    name = "imagenet_train",
    srcs = [
        "imagenet_train.py",
    ],
    deps = [
        ":imagenet_data",
        ":inception_train",
    ],
)

py_binary(
    name = "imagenet_distributed_train",
    srcs = [
        "imagenet_distributed_train.py",
    ],
    deps = [
        ":imagenet_data",
        ":inception_distributed_train",
    ],
)

py_binary(
    name = "flowers_train",
    srcs = [
        "flowers_train.py",
    ],
    deps = [
        ":flowers_data",
        ":inception_train",
    ],
)

py_library(
    name = "inception_train",
    srcs = [
        "inception_train.py",
    ],
    deps = [
        ":image_processing",
        ":inception",
    ],
)

py_library(
    name = "inception_distributed_train",
    srcs = [
        "inception_distributed_train.py",
    ],
    deps = [
        ":image_processing",
        ":inception",
    ],
)

py_binary(
    name = "build_image_data",
    srcs = ["data/build_image_data.py"],
)

sh_binary(
    name = "download_and_preprocess_flowers",
    srcs = ["data/download_and_preprocess_flowers.sh"],
    data = [
        ":build_image_data",
    ],
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
