# Tensorflow Object Detection API: Configuration protos.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

proto_library(
    name = "argmax_matcher_proto",
    srcs = ["argmax_matcher.proto"],
)

py_proto_library(
    name = "argmax_matcher_py_pb2",
    api_version = 2,
    deps = [":argmax_matcher_proto"],
)

proto_library(
    name = "bipartite_matcher_proto",
    srcs = ["bipartite_matcher.proto"],
)

py_proto_library(
    name = "bipartite_matcher_py_pb2",
    api_version = 2,
    deps = [":bipartite_matcher_proto"],
)

proto_library(
    name = "matcher_proto",
    srcs = ["matcher.proto"],
    deps = [
        ":argmax_matcher_proto",
        ":bipartite_matcher_proto",
    ],
)

py_proto_library(
    name = "matcher_py_pb2",
    api_version = 2,
    deps = [":matcher_proto"],
)

proto_library(
    name = "faster_rcnn_box_coder_proto",
    srcs = ["faster_rcnn_box_coder.proto"],
)

py_proto_library(
    name = "faster_rcnn_box_coder_py_pb2",
    api_version = 2,
    deps = [":faster_rcnn_box_coder_proto"],
)

proto_library(
    name = "mean_stddev_box_coder_proto",
    srcs = ["mean_stddev_box_coder.proto"],
)

py_proto_library(
    name = "mean_stddev_box_coder_py_pb2",
    api_version = 2,
    deps = [":mean_stddev_box_coder_proto"],
)

proto_library(
    name = "square_box_coder_proto",
    srcs = ["square_box_coder.proto"],
)

py_proto_library(
    name = "square_box_coder_py_pb2",
    api_version = 2,
    deps = [":square_box_coder_proto"],
)

proto_library(
    name = "box_coder_proto",
    srcs = ["box_coder.proto"],
    deps = [
        ":faster_rcnn_box_coder_proto",
        ":mean_stddev_box_coder_proto",
        ":square_box_coder_proto",
    ],
)

py_proto_library(
    name = "box_coder_py_pb2",
    api_version = 2,
    deps = [":box_coder_proto"],
)

proto_library(
    name = "grid_anchor_generator_proto",
    srcs = ["grid_anchor_generator.proto"],
)

py_proto_library(
    name = "grid_anchor_generator_py_pb2",
    api_version = 2,
    deps = [":grid_anchor_generator_proto"],
)

proto_library(
    name = "ssd_anchor_generator_proto",
    srcs = ["ssd_anchor_generator.proto"],
)

py_proto_library(
    name = "ssd_anchor_generator_py_pb2",
    api_version = 2,
    deps = [":ssd_anchor_generator_proto"],
)

proto_library(
    name = "anchor_generator_proto",
    srcs = ["anchor_generator.proto"],
    deps = [
        ":grid_anchor_generator_proto",
        ":ssd_anchor_generator_proto",
    ],
)

py_proto_library(
    name = "anchor_generator_py_pb2",
    api_version = 2,
    deps = [":anchor_generator_proto"],
)

proto_library(
    name = "input_reader_proto",
    srcs = ["input_reader.proto"],
)

py_proto_library(
    name = "input_reader_py_pb2",
    api_version = 2,
    deps = [":input_reader_proto"],
)

proto_library(
    name = "losses_proto",
    srcs = ["losses.proto"],
)

py_proto_library(
    name = "losses_py_pb2",
    api_version = 2,
    deps = [":losses_proto"],
)

proto_library(
    name = "optimizer_proto",
    srcs = ["optimizer.proto"],
)

py_proto_library(
    name = "optimizer_py_pb2",
    api_version = 2,
    deps = [":optimizer_proto"],
)

proto_library(
    name = "post_processing_proto",
    srcs = ["post_processing.proto"],
)

py_proto_library(
    name = "post_processing_py_pb2",
    api_version = 2,
    deps = [":post_processing_proto"],
)

proto_library(
    name = "hyperparams_proto",
    srcs = ["hyperparams.proto"],
)

py_proto_library(
    name = "hyperparams_py_pb2",
    api_version = 2,
    deps = [":hyperparams_proto"],
)

proto_library(
    name = "box_predictor_proto",
    srcs = ["box_predictor.proto"],
    deps = [":hyperparams_proto"],
)

py_proto_library(
    name = "box_predictor_py_pb2",
    api_version = 2,
    deps = [":box_predictor_proto"],
)

proto_library(
    name = "region_similarity_calculator_proto",
    srcs = ["region_similarity_calculator.proto"],
    deps = [],
)

py_proto_library(
    name = "region_similarity_calculator_py_pb2",
    api_version = 2,
    deps = [":region_similarity_calculator_proto"],
)

proto_library(
    name = "preprocessor_proto",
    srcs = ["preprocessor.proto"],
)

py_proto_library(
    name = "preprocessor_py_pb2",
    api_version = 2,
    deps = [":preprocessor_proto"],
)

proto_library(
    name = "train_proto",
    srcs = ["train.proto"],
    deps = [
        ":optimizer_proto",
        ":preprocessor_proto",
    ],
)

py_proto_library(
    name = "train_py_pb2",
    api_version = 2,
    deps = [":train_proto"],
)

proto_library(
    name = "eval_proto",
    srcs = ["eval.proto"],
)

py_proto_library(
    name = "eval_py_pb2",
    api_version = 2,
    deps = [":eval_proto"],
)

proto_library(
    name = "image_resizer_proto",
    srcs = ["image_resizer.proto"],
)

py_proto_library(
    name = "image_resizer_py_pb2",
    api_version = 2,
    deps = [":image_resizer_proto"],
)

proto_library(
    name = "faster_rcnn_proto",
    srcs = ["faster_rcnn.proto"],
    deps = [
        ":box_predictor_proto",
        "//object_detection/protos:anchor_generator_proto",
        "//object_detection/protos:hyperparams_proto",
        "//object_detection/protos:image_resizer_proto",
        "//object_detection/protos:losses_proto",
        "//object_detection/protos:post_processing_proto",
    ],
)

proto_library(
    name = "ssd_proto",
    srcs = ["ssd.proto"],
    deps = [
        ":anchor_generator_proto",
        ":box_coder_proto",
        ":box_predictor_proto",
        ":hyperparams_proto",
        ":image_resizer_proto",
        ":losses_proto",
        ":matcher_proto",
        ":post_processing_proto",
        ":region_similarity_calculator_proto",
    ],
)

proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
    deps = [
        ":faster_rcnn_proto",
        ":ssd_proto",
    ],
)

py_proto_library(
    name = "model_py_pb2",
    api_version = 2,
    deps = [":model_proto"],
)

proto_library(
    name = "pipeline_proto",
    srcs = ["pipeline.proto"],
    deps = [
        ":eval_proto",
        ":input_reader_proto",
        ":model_proto",
        ":train_proto",
    ],
)

py_proto_library(
    name = "pipeline_py_pb2",
    api_version = 2,
    deps = [":pipeline_proto"],
)

proto_library(
    name = "string_int_label_map_proto",
    srcs = ["string_int_label_map.proto"],
)

py_proto_library(
    name = "string_int_label_map_py_pb2",
    api_version = 2,
    deps = [":string_int_label_map_proto"],
)
