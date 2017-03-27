# Domain Adaptation Scenarios Datasets

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
    name = "dataset_factory",
    srcs = ["dataset_factory.py"],
    deps = [
        ":mnist_m",
        "//slim:mnist",
    ],
)

py_binary(
    name = "mnist_m",
    srcs = ["mnist_m.py"],
    deps = [
        "//slim:dataset_utils",
    ],
)
