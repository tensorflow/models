# Files for working with skeletons and random data

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "random_data_provider",
    srcs = ["random_data_provider.py"],
    deps = [
        "//base:config",
    ],
)

py_binary(
    name = "nn_trainer",
    srcs = ["nn_trainer.py"],
    deps = [
        ":random_data_provider",
        "//base:nn_trainer",
    ],
)

py_binary(
    name = "rf_trainer",
    srcs = ["rf_trainer.py"],
    deps = [
        ":random_data_provider",
        "//base:rf_trainer",
    ],
)
