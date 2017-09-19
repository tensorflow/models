package(default_visibility = ["//visibility:public"])

# Libraries
# =========
py_library(
    name = "coordinatewise_rnn",
    srcs = ["coordinatewise_rnn.py"],
    deps = [
        ":trainable_optimizer",
        ":utils",
    ],
)

py_library(
    name = "global_learning_rate",
    srcs = ["global_learning_rate.py"],
    deps = [
        ":trainable_optimizer",
    ],
)

py_library(
    name = "hierarchical_rnn",
    srcs = ["hierarchical_rnn.py"],
    deps = [
        ":rnn_cells",
        ":trainable_optimizer",
        ":utils",
    ],
)

py_library(
    name = "learning_rate_schedule",
    srcs = ["learning_rate_schedule.py"],
    deps = [
        ":trainable_optimizer",
    ],
)

py_library(
    name = "rnn_cells",
    srcs = ["rnn_cells.py"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "trainable_adam",
    srcs = ["trainable_adam.py"],
    deps = [
        ":trainable_optimizer",
        ":utils",
    ],
)

py_library(
    name = "trainable_optimizer",
    srcs = ["trainable_optimizer.py"],
    deps = [
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
    ],
)
