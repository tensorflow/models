licenses(["notice"])  # Apache 2.0

# Binaries
# ==============================================================================
py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        ":graphs",
        # google3 file dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "train_classifier",
    srcs = ["train_classifier.py"],
    deps = [
        ":graphs",
        ":train_utils",
        # google3 file dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "pretrain",
    srcs = [
        "pretrain.py",
    ],
    deps = [
        ":graphs",
        ":train_utils",
        # google3 file dep,
        # tensorflow dep,
    ],
)

# Libraries
# ==============================================================================
py_library(
    name = "graphs",
    srcs = ["graphs.py"],
    deps = [
        ":adversarial_losses",
        ":inputs",
        ":layers",
        # tensorflow dep,
    ],
)

py_library(
    name = "adversarial_losses",
    srcs = ["adversarial_losses.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs.py"],
    deps = [
        # tensorflow dep,
        "//adversarial_text/data:data_utils",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "train_utils",
    srcs = ["train_utils.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

# Tests
# ==============================================================================
py_test(
    name = "graphs_test",
    size = "large",
    srcs = ["graphs_test.py"],
    deps = [
        ":graphs",
        # tensorflow dep,
        "//adversarial_text/data:data_utils",
    ],
)
