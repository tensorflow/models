package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//cognitive_planning/...",
    ],
)

py_binary(
    name = "train_supervised_active_vision",
    srcs = [
        "train_supervised_active_vision.py",
    ],
)
