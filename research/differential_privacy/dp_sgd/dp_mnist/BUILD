package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//differential_privacy/...",
    ],
)

py_binary(
    name = "dp_mnist",
    srcs = [
        "dp_mnist.py",
    ],
    deps = [
        "//differential_privacy/dp_sgd/dp_optimizer",
        "//differential_privacy/dp_sgd/dp_optimizer:dp_pca",
        "//differential_privacy/dp_sgd/dp_optimizer:utils",
    ],
)
