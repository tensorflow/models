# Tensorflow Object Detection API: Meta-architectures.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "ssd_meta_arch",
    srcs = ["ssd_meta_arch.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:box_predictor",
        "//tensorflow_models/object_detection/core:model",
        "//tensorflow_models/object_detection/core:target_assigner",
        "//tensorflow_models/object_detection/utils:shape_utils",
    ],
)

py_test(
    name = "ssd_meta_arch_test",
    srcs = ["ssd_meta_arch_test.py"],
    deps = [
        ":ssd_meta_arch",
        "//tensorflow",
        "//tensorflow/python:training",
        "//tensorflow_models/object_detection/core:anchor_generator",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:losses",
        "//tensorflow_models/object_detection/core:post_processing",
        "//tensorflow_models/object_detection/core:region_similarity_calculator",
        "//tensorflow_models/object_detection/utils:test_utils",
    ],
)

py_library(
    name = "faster_rcnn_meta_arch",
    srcs = [
        "faster_rcnn_meta_arch.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/anchor_generators:grid_anchor_generator",
        "//tensorflow_models/object_detection/core:balanced_positive_negative_sampler",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:box_list_ops",
        "//tensorflow_models/object_detection/core:box_predictor",
        "//tensorflow_models/object_detection/core:losses",
        "//tensorflow_models/object_detection/core:model",
        "//tensorflow_models/object_detection/core:post_processing",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/core:target_assigner",
        "//tensorflow_models/object_detection/utils:ops",
        "//tensorflow_models/object_detection/utils:shape_utils",
    ],
)

py_library(
    name = "faster_rcnn_meta_arch_test_lib",
    srcs = [
        "faster_rcnn_meta_arch_test_lib.py",
    ],
    deps = [
        ":faster_rcnn_meta_arch",
        "//tensorflow",
        "//tensorflow_models/object_detection/anchor_generators:grid_anchor_generator",
        "//tensorflow_models/object_detection/builders:box_predictor_builder",
        "//tensorflow_models/object_detection/builders:hyperparams_builder",
        "//tensorflow_models/object_detection/builders:post_processing_builder",
        "//tensorflow_models/object_detection/core:losses",
        "//tensorflow_models/object_detection/protos:box_predictor_py_pb2",
        "//tensorflow_models/object_detection/protos:hyperparams_py_pb2",
        "//tensorflow_models/object_detection/protos:post_processing_py_pb2",
    ],
)

py_test(
    name = "faster_rcnn_meta_arch_test",
    srcs = ["faster_rcnn_meta_arch_test.py"],
    deps = [
        ":faster_rcnn_meta_arch_test_lib",
    ],
)

py_library(
    name = "rfcn_meta_arch",
    srcs = ["rfcn_meta_arch.py"],
    deps = [
        ":faster_rcnn_meta_arch",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_predictor",
        "//tensorflow_models/object_detection/utils:ops",
    ],
)

py_test(
    name = "rfcn_meta_arch_test",
    srcs = ["rfcn_meta_arch_test.py"],
    deps = [
        ":faster_rcnn_meta_arch_test_lib",
        ":rfcn_meta_arch",
        "//tensorflow",
    ],
)
