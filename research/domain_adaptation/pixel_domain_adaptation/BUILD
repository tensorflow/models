# Description:
#   Contains code for domain-adaptation style transfer.

package(
    default_visibility = [
        ":internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//domain_adaptation/...",
    ],
)

py_library(
    name = "pixelda_preprocess",
    srcs = ["pixelda_preprocess.py"],
    deps = [
        
    ],
)

py_test(
    name = "pixelda_preprocess_test",
    srcs = ["pixelda_preprocess_test.py"],
    deps = [
        ":pixelda_preprocess",
        
    ],
)

py_library(
    name = "pixelda_model",
    srcs = [
        "pixelda_model.py",
        "pixelda_task_towers.py",
        "hparams.py",
    ],
    deps = [
        
    ],
)

py_library(
    name = "pixelda_utils",
    srcs = ["pixelda_utils.py"],
    deps = [
        
    ],
)

py_library(
    name = "pixelda_losses",
    srcs = ["pixelda_losses.py"],
    deps = [
        
    ],
)

py_binary(
    name = "pixelda_train",
    srcs = ["pixelda_train.py"],
    deps = [
        ":pixelda_losses",
        ":pixelda_model",
        ":pixelda_preprocess",
        ":pixelda_utils",
        
        "//domain_adaptation/datasets:dataset_factory",
    ],
)

py_binary(
    name = "pixelda_eval",
    srcs = ["pixelda_eval.py"],
    deps = [
        ":pixelda_losses",
        ":pixelda_model",
        ":pixelda_preprocess",
        ":pixelda_utils",
        
        "//domain_adaptation/datasets:dataset_factory",
    ],
)
