package(default_visibility = ["//visibility:public"])

# Libraries
# =====

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    deps = [
    ],
)

py_library(
    name = "model_adapter",
    srcs = ["model_adapter.py"],
    deps = [
        ":problem_generator",
    ],
)

py_library(
    name = "problem_generator",
    srcs = ["problem_generator.py"],
    deps = [
        ":problem_spec",
    ],
)

py_library(
    name = "problem_sets",
    srcs = ["problem_sets.py"],
    deps = [
        ":datasets",
        ":model_adapter",
        ":problem_generator",
    ],
)

py_library(
    name = "problem_spec",
    srcs = ["problem_spec.py"],
    deps = [],
)
