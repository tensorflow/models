package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//differential_privacy/...",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    deps = [
    ],
)

py_library(
    name = "dp_pca",
    srcs = [
        "dp_pca.py",
    ],
    deps = [
    ],
)

py_library(
    name = "dp_optimizer",
    srcs = [
        "dp_optimizer.py",
        "sanitizer.py",
    ],
    deps = [
        ":utils",
        "//differential_privacy/dp_sgd/per_example_gradients",
        "//differential_privacy/privacy_accountant/tf:accountant",
    ],
)

