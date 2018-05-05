py_library(
    name = "input_generator",
    srcs = ["input_generator.py"],
    deps = [
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
    ],
)

# Defines the Rotator model here
py_library(
    name = "model_rotator",
    srcs = ["model_rotator.py"],
    deps = [
        ":input_generator",
        ":losses",
        ":metrics",
        ":utils",
        "//nets:deeprotator_factory",
    ],
)

# Defines the Im2vox model here
py_library(
    name = "model_voxel_generation",
    srcs = ["model_voxel_generation.py"],
    deps = [
        ":input_generator",
        "//nets:im2vox_factory",
    ],
)

py_library(
    name = "model_ptn",
    srcs = ["model_ptn.py"],
    deps = [
        ":losses",
        ":metrics",
        ":model_voxel_generation",
        ":utils",
        "//nets:im2vox_factory",
    ],
)

py_binary(
    name = "train_ptn",
    srcs = ["train_ptn.py"],
    deps = [
        ":model_ptn",
    ],
)

py_binary(
    name = "eval_ptn",
    srcs = ["eval_ptn.py"],
    deps = [
        ":model_ptn",
    ],
)

py_binary(
    name = "pretrain_rotator",
    srcs = ["pretrain_rotator.py"],
    deps = [
        ":model_rotator",
    ],
)

py_binary(
    name = "eval_rotator",
    srcs = ["eval_rotator.py"],
    deps = [
        ":model_rotator",
    ],
)
