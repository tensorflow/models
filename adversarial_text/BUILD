# Binaries
# ==============================================================================
py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        ":graphs",
    ],
)

py_binary(
    name = "train_classifier",
    srcs = ["train_classifier.py"],
    deps = [
        ":graphs",
        ":train_utils",
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
    ],
)

py_library(
    name = "adversarial_losses",
    srcs = ["adversarial_losses.py"],
)

py_library(
    name = "inputs",
    srcs = ["inputs.py"],
    deps = [
        "//adversarial_text/data:data_utils",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
)

py_library(
    name = "train_utils",
    srcs = ["train_utils.py"],
)

# Tests
# ==============================================================================
py_test(
    name = "graphs_test",
    size = "large",
    srcs = ["graphs_test.py"],
    deps = [
        ":graphs",
        "//adversarial_text/data:data_utils",
    ],
)
