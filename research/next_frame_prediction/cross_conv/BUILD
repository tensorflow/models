licenses(["notice"])  # Apache 2.0

package_group(
    name = "internal",
    packages = [
        "//next_frame_prediction/...",
    ],
)

package(default_visibility = [":internal"])

py_library(
    name = "model",
    srcs = ["model.py"],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":model",
        ":reader",
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":model",
        ":reader",
    ],
)

py_binary(
    name = "example_gen",
    srcs = ["example_gen.py"],
)

py_binary(
    name = "sprites_gen",
    srcs = ["sprites_gen.py"],
)
