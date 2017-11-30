# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0
py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        ":trainer",
        "//tensorflow",
        "//tensorflow_models/object_detection/builders:input_reader_builder",
        "//tensorflow_models/object_detection/builders:model_builder",
        "//tensorflow_models/object_detection/utils:config_util",
    ],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/builders:optimizer_builder",
        "//tensorflow_models/object_detection/builders:preprocessor_builder",
        "//tensorflow_models/object_detection/core:batcher",
        "//tensorflow_models/object_detection/core:preprocessor",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/utils:ops",
        "//tensorflow_models/object_detection/utils:variables_helper",
        "//tensorflow_models/slim:model_deploy",
    ],
)

py_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    deps = [
        ":trainer",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:losses",
        "//tensorflow_models/object_detection/core:model",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/protos:train_py_pb2",
    ],
)

py_library(
    name = "eval_util",
    srcs = [
        "eval_util.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:box_list_ops",
        "//tensorflow_models/object_detection/core:keypoint_ops",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/utils:label_map_util",
        "//tensorflow_models/object_detection/utils:ops",
        "//tensorflow_models/object_detection/utils:visualization_utils",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection:eval_util",
        "//tensorflow_models/object_detection/core:prefetcher",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/protos:eval_py_pb2",
        "//tensorflow_models/object_detection/utils:object_detection_evaluation",
    ],
)

py_binary(
    name = "eval",
    srcs = [
        "eval.py",
    ],
    deps = [
        ":evaluator",
        "//tensorflow",
        "//tensorflow_models/object_detection/builders:input_reader_builder",
        "//tensorflow_models/object_detection/builders:model_builder",
        "//tensorflow_models/object_detection/utils:config_util",
        "//tensorflow_models/object_detection/utils:label_map_util",
    ],
)

py_library(
    name = "exporter",
    srcs = [
        "exporter.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/python/tools:freeze_graph_lib",
        "//tensorflow_models/object_detection/builders:model_builder",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/data_decoders:tf_example_decoder",
    ],
)

py_test(
    name = "exporter_test",
    srcs = [
        "exporter_test.py",
    ],
    deps = [
        ":exporter",
        "//tensorflow",
        "//tensorflow_models/object_detection/builders:model_builder",
        "//tensorflow_models/object_detection/core:model",
        "//tensorflow_models/object_detection/protos:pipeline_py_pb2",
    ],
)

py_binary(
    name = "export_inference_graph",
    srcs = [
        "export_inference_graph.py",
    ],
    deps = [
        ":exporter",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:pipeline_py_pb2",
    ],
)
