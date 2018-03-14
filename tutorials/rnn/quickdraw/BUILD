# Description:
# Example classification model on Quick, Draw! dataset.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "train_model",
    srcs = [
        "train_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "create_dataset",
    srcs = [
        "create_dataset.py",
    ],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//third_party/tensorflow:__subpackages__"],
)
