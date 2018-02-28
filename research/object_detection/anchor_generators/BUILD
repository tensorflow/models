# Tensorflow Object Detection API: Anchor Generator implementations.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0
py_library(
    name = "grid_anchor_generator",
    srcs = [
        "grid_anchor_generator.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:anchor_generator",
        "//tensorflow/models/research/object_detection/core:box_list",
        "//tensorflow/models/research/object_detection/utils:ops",
    ],
)

py_test(
    name = "grid_anchor_generator_test",
    srcs = [
        "grid_anchor_generator_test.py",
    ],
    deps = [
        ":grid_anchor_generator",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:test_case",
    ],
)

py_library(
    name = "multiple_grid_anchor_generator",
    srcs = [
        "multiple_grid_anchor_generator.py",
    ],
    deps = [
        ":grid_anchor_generator",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:anchor_generator",
        "//tensorflow/models/research/object_detection/core:box_list_ops",
    ],
)

py_test(
    name = "multiple_grid_anchor_generator_test",
    srcs = [
        "multiple_grid_anchor_generator_test.py",
    ],
    deps = [
        ":multiple_grid_anchor_generator",
        "//numpy",
        "//tensorflow/models/research/object_detection/utils:test_case",
    ],
)

py_library(
    name = "multiscale_grid_anchor_generator",
    srcs = [
        "multiscale_grid_anchor_generator.py",
    ],
    deps = [
        ":grid_anchor_generator",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:anchor_generator",
        "//tensorflow/models/research/object_detection/core:box_list_ops",
    ],
)

py_test(
    name = "multiscale_grid_anchor_generator_test",
    srcs = [
        "multiscale_grid_anchor_generator_test.py",
    ],
    deps = [
        ":multiscale_grid_anchor_generator",
        "//numpy",
        "//tensorflow/models/research/object_detection/utils:test_case",
    ],
)
