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
    name = "accountant",
    srcs = [
        "accountant.py",
    ],
    deps = [
        "//differential_privacy/dp_sgd/dp_optimizer:utils",
    ],
)

