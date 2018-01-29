# Tensorflow Object Detection API: Core.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])
# Apache 2.0

py_library(
    name = "batcher",
    srcs = ["batcher.py"],
    deps = [
        ":prefetcher",
        ":preprocessor",
        ":standard_fields",
        "//tensorflow",
    ],
)

py_test(
    name = "batcher_test",
    srcs = ["batcher_test.py"],
    deps = [
        ":batcher",
        "//tensorflow",
    ],
)

py_library(
    name = "box_list",
    srcs = [
        "box_list.py",
    ],
    deps = [
        "//tensorflow",
    ],
)

py_test(
    name = "box_list_test",
    srcs = ["box_list_test.py"],
    deps = [
        ":box_list",
    ],
)

py_library(
    name = "box_list_ops",
    srcs = [
        "box_list_ops.py",
    ],
    deps = [
        ":box_list",
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:shape_utils",
    ],
)

py_test(
    name = "box_list_ops_test",
    srcs = ["box_list_ops_test.py"],
    deps = [
        ":box_list",
        ":box_list_ops",
    ],
)

py_library(
    name = "box_coder",
    srcs = [
        "box_coder.py",
    ],
    deps = [
        "//tensorflow",
    ],
)

py_test(
    name = "box_coder_test",
    srcs = [
        "box_coder_test.py",
    ],
    deps = [
        ":box_coder",
        ":box_list",
        "//tensorflow",
    ],
)

py_library(
    name = "keypoint_ops",
    srcs = [
        "keypoint_ops.py",
    ],
    deps = [
        "//tensorflow",
    ],
)

py_test(
    name = "keypoint_ops_test",
    srcs = ["keypoint_ops_test.py"],
    deps = [
        ":keypoint_ops",
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [
        ":box_list",
        ":box_list_ops",
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:ops",
    ],
)

py_library(
    name = "matcher",
    srcs = [
        "matcher.py",
    ],
    deps = [
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":standard_fields",
    ],
)

py_test(
    name = "matcher_test",
    srcs = [
        "matcher_test.py",
    ],
    deps = [
        ":matcher",
        "//tensorflow",
    ],
)

py_library(
    name = "prefetcher",
    srcs = ["prefetcher.py"],
    deps = ["//tensorflow"],
)

py_library(
    name = "preprocessor",
    srcs = [
        "preprocessor.py",
    ],
    deps = [
        ":box_list",
        ":box_list_ops",
        ":keypoint_ops",
        ":standard_fields",
        "//tensorflow",
    ],
)

py_test(
    name = "preprocessor_test",
    srcs = [
        "preprocessor_test.py",
    ],
    deps = [
        ":preprocessor",
        "//tensorflow",
    ],
)

py_test(
    name = "losses_test",
    srcs = ["losses_test.py"],
    deps = [
        ":box_list",
        ":losses",
        ":matcher",
        "//tensorflow",
    ],
)

py_test(
    name = "prefetcher_test",
    srcs = ["prefetcher_test.py"],
    deps = [
        ":prefetcher",
        "//tensorflow",
    ],
)

py_library(
    name = "standard_fields",
    srcs = [
        "standard_fields.py",
    ],
)

py_library(
    name = "post_processing",
    srcs = ["post_processing.py"],
    deps = [
        ":box_list",
        ":box_list_ops",
        ":standard_fields",
        "//tensorflow",
    ],
)

py_test(
    name = "post_processing_test",
    srcs = ["post_processing_test.py"],
    deps = [
        ":box_list",
        ":box_list_ops",
        ":post_processing",
        "//tensorflow",
    ],
)

py_library(
    name = "target_assigner",
    srcs = [
        "target_assigner.py",
    ],
    deps = [
        ":box_list",
        ":box_list_ops",
        ":matcher",
        ":region_similarity_calculator",
        "//tensorflow",
        "//tensorflow_models/object_detection/box_coders:faster_rcnn_box_coder",
        "//tensorflow_models/object_detection/box_coders:mean_stddev_box_coder",
        "//tensorflow_models/object_detection/core:box_coder",
        "//tensorflow_models/object_detection/matchers:argmax_matcher",
        "//tensorflow_models/object_detection/matchers:bipartite_matcher",
    ],
)

py_test(
    name = "target_assigner_test",
    size = "large",
    timeout = "long",
    srcs = ["target_assigner_test.py"],
    deps = [
        ":box_list",
        ":region_similarity_calculator",
        ":target_assigner",
        "//tensorflow",
        "//tensorflow_models/object_detection/box_coders:mean_stddev_box_coder",
        "//tensorflow_models/object_detection/matchers:bipartite_matcher",
    ],
)

py_library(
    name = "data_decoder",
    srcs = ["data_decoder.py"],
)

py_library(
    name = "data_parser",
    srcs = ["data_parser.py"],
)

py_library(
    name = "box_predictor",
    srcs = ["box_predictor.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:ops",
        "//tensorflow_models/object_detection/utils:shape_utils",
        "//tensorflow_models/object_detection/utils:static_shape",
    ],
)

py_test(
    name = "box_predictor_test",
    srcs = ["box_predictor_test.py"],
    deps = [
        ":box_predictor",
        "//tensorflow",
        "//tensorflow_models/object_detection/builders:hyperparams_builder",
        "//tensorflow_models/object_detection/protos:hyperparams_py_pb2",
    ],
)

py_library(
    name = "region_similarity_calculator",
    srcs = [
        "region_similarity_calculator.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list_ops",
    ],
)

py_test(
    name = "region_similarity_calculator_test",
    srcs = [
        "region_similarity_calculator_test.py",
    ],
    deps = [
        ":region_similarity_calculator",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)

py_library(
    name = "anchor_generator",
    srcs = [
        "anchor_generator.py",
    ],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "minibatch_sampler",
    srcs = [
        "minibatch_sampler.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:ops",
    ],
)

py_test(
    name = "minibatch_sampler_test",
    srcs = [
        "minibatch_sampler_test.py",
    ],
    deps = [
        ":minibatch_sampler",
        "//tensorflow",
    ],
)

py_library(
    name = "balanced_positive_negative_sampler",
    srcs = [
        "balanced_positive_negative_sampler.py",
    ],
    deps = [
        ":minibatch_sampler",
        "//tensorflow",
    ],
)

py_test(
    name = "balanced_positive_negative_sampler_test",
    srcs = [
        "balanced_positive_negative_sampler_test.py",
    ],
    deps = [
        ":balanced_positive_negative_sampler",
        "//tensorflow",
    ],
)
