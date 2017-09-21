package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//lm_1b/...",
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
)

py_binary(
    name = "lm_1b_eval",
    srcs = [
        "lm_1b_eval.py",
    ],
    deps = [
        ":data_utils",
    ],
)
