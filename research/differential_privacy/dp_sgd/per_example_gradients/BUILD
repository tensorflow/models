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
    name = "per_example_gradients",
    srcs = [
        "per_example_gradients.py",
    ],
    deps = [
    ],
)

