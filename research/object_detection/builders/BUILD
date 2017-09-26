# Tensorflow Object Detection API: component builders.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0
py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    deps = [
        ":anchor_generator_builder",
        ":box_coder_builder",
        ":box_predictor_builder",
        ":hyperparams_builder",
        ":image_resizer_builder",
        ":losses_builder",
        ":matcher_builder",
        ":post_processing_builder",
        ":region_similarity_calculator_builder",
        "//tensorflow_models/object_detection/core:box_predictor",
        "//tensorflow_models/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//tensorflow_models/object_detection/meta_architectures:rfcn_meta_arch",
        "//tensorflow_models/object_detection/meta_architectures:ssd_meta_arch",
        "//tensorflow_models/object_detection/models:faster_rcnn_inception_resnet_v2_feature_extractor",
        "//tensorflow_models/object_detection/models:faster_rcnn_resnet_v1_feature_extractor",
        "//tensorflow_models/object_detection/models:ssd_inception_v2_feature_extractor",
        "//tensorflow_models/object_detection/models:ssd_mobilenet_v1_feature_extractor",
        "//tensorflow_models/object_detection/protos:model_py_pb2",
    ],
)

py_test(
    name = "model_builder_test",
    srcs = ["model_builder_test.py"],
    deps = [
        ":model_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//tensorflow_models/object_detection/meta_architectures:ssd_meta_arch",
        "//tensorflow_models/object_detection/models:ssd_inception_v2_feature_extractor",
        "//tensorflow_models/object_detection/models:ssd_mobilenet_v1_feature_extractor",
        "//tensorflow_models/object_detection/protos:model_py_pb2",
    ],
)

py_library(
    name = "matcher_builder",
    srcs = ["matcher_builder.py"],
    deps = [
        "//tensorflow_models/object_detection/matchers:argmax_matcher",
        "//tensorflow_models/object_detection/matchers:bipartite_matcher",
        "//tensorflow_models/object_detection/protos:matcher_py_pb2",
    ],
)

py_test(
    name = "matcher_builder_test",
    srcs = ["matcher_builder_test.py"],
    deps = [
        ":matcher_builder",
        "//tensorflow_models/object_detection/matchers:argmax_matcher",
        "//tensorflow_models/object_detection/matchers:bipartite_matcher",
        "//tensorflow_models/object_detection/protos:matcher_py_pb2",
    ],
)

py_library(
    name = "box_coder_builder",
    srcs = ["box_coder_builder.py"],
    deps = [
        "//tensorflow_models/object_detection/box_coders:faster_rcnn_box_coder",
        "//tensorflow_models/object_detection/box_coders:mean_stddev_box_coder",
        "//tensorflow_models/object_detection/box_coders:square_box_coder",
        "//tensorflow_models/object_detection/protos:box_coder_py_pb2",
    ],
)

py_test(
    name = "box_coder_builder_test",
    srcs = ["box_coder_builder_test.py"],
    deps = [
        ":box_coder_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/box_coders:faster_rcnn_box_coder",
        "//tensorflow_models/object_detection/box_coders:mean_stddev_box_coder",
        "//tensorflow_models/object_detection/box_coders:square_box_coder",
        "//tensorflow_models/object_detection/protos:box_coder_py_pb2",
    ],
)

py_library(
    name = "anchor_generator_builder",
    srcs = ["anchor_generator_builder.py"],
    deps = [
        "//tensorflow_models/object_detection/anchor_generators:grid_anchor_generator",
        "//tensorflow_models/object_detection/anchor_generators:multiple_grid_anchor_generator",
        "//tensorflow_models/object_detection/protos:anchor_generator_py_pb2",
    ],
)

py_test(
    name = "anchor_generator_builder_test",
    srcs = ["anchor_generator_builder_test.py"],
    deps = [
        ":anchor_generator_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/anchor_generators:grid_anchor_generator",
        "//tensorflow_models/object_detection/anchor_generators:multiple_grid_anchor_generator",
        "//tensorflow_models/object_detection/protos:anchor_generator_py_pb2",
    ],
)

py_library(
    name = "input_reader_builder",
    srcs = ["input_reader_builder.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/data_decoders:tf_example_decoder",
        "//tensorflow_models/object_detection/protos:input_reader_py_pb2",
    ],
)

py_test(
    name = "input_reader_builder_test",
    srcs = [
        "input_reader_builder_test.py",
    ],
    deps = [
        ":input_reader_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/protos:input_reader_py_pb2",
    ],
)

py_library(
    name = "losses_builder",
    srcs = ["losses_builder.py"],
    deps = [
        "//tensorflow_models/object_detection/core:losses",
        "//tensorflow_models/object_detection/protos:losses_py_pb2",
    ],
)

py_test(
    name = "losses_builder_test",
    srcs = ["losses_builder_test.py"],
    deps = [
        ":losses_builder",
        "//tensorflow_models/object_detection/core:losses",
        "//tensorflow_models/object_detection/protos:losses_py_pb2",
    ],
)

py_library(
    name = "optimizer_builder",
    srcs = ["optimizer_builder.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:learning_schedules",
    ],
)

py_test(
    name = "optimizer_builder_test",
    srcs = ["optimizer_builder_test.py"],
    deps = [
        ":optimizer_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:optimizer_py_pb2",
    ],
)

py_library(
    name = "post_processing_builder",
    srcs = ["post_processing_builder.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:post_processing",
        "//tensorflow_models/object_detection/protos:post_processing_py_pb2",
    ],
)

py_test(
    name = "post_processing_builder_test",
    srcs = ["post_processing_builder_test.py"],
    deps = [
        ":post_processing_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:post_processing_py_pb2",
    ],
)

py_library(
    name = "hyperparams_builder",
    srcs = ["hyperparams_builder.py"],
    deps = [
        "//tensorflow_models/object_detection/protos:hyperparams_py_pb2",
    ],
)

py_test(
    name = "hyperparams_builder_test",
    srcs = ["hyperparams_builder_test.py"],
    deps = [
        ":hyperparams_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:hyperparams_py_pb2",
    ],
)

py_library(
    name = "box_predictor_builder",
    srcs = ["box_predictor_builder.py"],
    deps = [
        ":hyperparams_builder",
        "//tensorflow_models/object_detection/core:box_predictor",
        "//tensorflow_models/object_detection/protos:box_predictor_py_pb2",
    ],
)

py_test(
    name = "box_predictor_builder_test",
    srcs = ["box_predictor_builder_test.py"],
    deps = [
        ":box_predictor_builder",
        ":hyperparams_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:box_predictor_py_pb2",
        "//tensorflow_models/object_detection/protos:hyperparams_py_pb2",
    ],
)

py_library(
    name = "region_similarity_calculator_builder",
    srcs = ["region_similarity_calculator_builder.py"],
    deps = [
        "//tensorflow_models/object_detection/core:region_similarity_calculator",
        "//tensorflow_models/object_detection/protos:region_similarity_calculator_py_pb2",
    ],
)

py_test(
    name = "region_similarity_calculator_builder_test",
    srcs = ["region_similarity_calculator_builder_test.py"],
    deps = [
        ":region_similarity_calculator_builder",
        "//tensorflow",
    ],
)

py_library(
    name = "preprocessor_builder",
    srcs = ["preprocessor_builder.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:preprocessor",
        "//tensorflow_models/object_detection/protos:preprocessor_py_pb2",
    ],
)

py_test(
    name = "preprocessor_builder_test",
    srcs = [
        "preprocessor_builder_test.py",
    ],
    deps = [
        ":preprocessor_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:preprocessor",
        "//tensorflow_models/object_detection/protos:preprocessor_py_pb2",
    ],
)

py_library(
    name = "image_resizer_builder",
    srcs = ["image_resizer_builder.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:preprocessor",
        "//tensorflow_models/object_detection/protos:image_resizer_py_pb2",
    ],
)

py_test(
    name = "image_resizer_builder_test",
    srcs = ["image_resizer_builder_test.py"],
    deps = [
        ":image_resizer_builder",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:image_resizer_py_pb2",
    ],
)
