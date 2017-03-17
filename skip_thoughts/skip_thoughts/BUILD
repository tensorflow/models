package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//skip_thoughts/...",
    ],
)

py_library(
    name = "configuration",
    srcs = ["configuration.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "skip_thoughts_model",
    srcs = ["skip_thoughts_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//skip_thoughts/ops:gru_cell",
        "//skip_thoughts/ops:input_ops",
    ],
)

py_test(
    name = "skip_thoughts_model_test",
    size = "large",
    srcs = ["skip_thoughts_model_test.py"],
    deps = [
        ":configuration",
        ":skip_thoughts_model",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":skip_thoughts_model",
    ],
)

py_binary(
    name = "track_perplexity",
    srcs = ["track_perplexity.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":skip_thoughts_model",
    ],
)

py_binary(
    name = "vocabulary_expansion",
    srcs = ["vocabulary_expansion.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "skip_thoughts_encoder",
    srcs = ["skip_thoughts_encoder.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skip_thoughts_model",
        "//skip_thoughts/data:special_words",
    ],
)

py_library(
    name = "encoder_manager",
    srcs = ["encoder_manager.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":skip_thoughts_encoder",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":encoder_manager",
        "//skip_thoughts:configuration",
    ],
)

