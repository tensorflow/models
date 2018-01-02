package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//third_party/tensorflow_models/...",
    ],
)

py_binary(
    name = "gaussian_moments",
    srcs = [
        "gaussian_moments.py",
    ],
    deps = [
    ],
)
