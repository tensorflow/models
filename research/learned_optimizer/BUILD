# Learning to Optimize Learning (LOL)

package(default_visibility = ["//visibility:public"])

# Libraries
# =========

py_library(
    name = "metaopt",
    srcs = ["metaopt.py"],
    deps = [
        "//learned_optimizer/problems:datasets",
        "//learned_optimizer/problems:problem_generator",
    ],
)

# Binaries
# ========
py_binary(
    name = "metarun",
    srcs = ["metarun.py"],
    deps = [
        ":metaopt",
        "//learned_optimizer/optimizer:coordinatewise_rnn",
        "//learned_optimizer/optimizer:global_learning_rate",
        "//learned_optimizer/optimizer:hierarchical_rnn",
        "//learned_optimizer/optimizer:learning_rate_schedule",
        "//learned_optimizer/optimizer:trainable_adam",
        "//learned_optimizer/problems:problem_sets",
        "//learned_optimizer/problems:problem_spec",
    ],
)

