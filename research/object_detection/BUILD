# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "inputs",
    srcs = [
        "inputs.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection:trainer",
        "//tensorflow/models/research/object_detection/builders:dataset_builder",
        "//tensorflow/models/research/object_detection/builders:preprocessor_builder",
        "//tensorflow/models/research/object_detection/protos:input_reader_py_pb2",
        "//tensorflow/models/research/object_detection/protos:train_py_pb2",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
        "//tensorflow/models/research/object_detection/utils:ops",
    ],
)

py_test(
    name = "inputs_test",
    srcs = [
        "inputs_test.py",
    ],
    data = [
        "//tensorflow/models/research/object_detection/data:pet_label_map.pbtxt",
        "//tensorflow/models/research/object_detection/samples/configs:faster_rcnn_resnet50_pets.config",
        "//tensorflow/models/research/object_detection/samples/configs:ssd_inception_v2_pets.config",
        "//tensorflow/models/research/object_detection/test_data:pets_examples.record",
    ],
    deps = [
        ":inputs",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/utils:config_util",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        ":trainer",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/builders:dataset_builder",
        "//tensorflow/models/research/object_detection/builders:model_builder",
        "//tensorflow/models/research/object_detection/utils:config_util",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
    ],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/builders:optimizer_builder",
        "//tensorflow/models/research/object_detection/builders:preprocessor_builder",
        "//tensorflow/models/research/object_detection/core:batcher",
        "//tensorflow/models/research/object_detection/core:preprocessor",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:variables_helper",
        "//third_party/tensorflow_models/slim:model_deploy",
    ],
)

py_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    deps = [
        ":trainer",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:losses",
        "//tensorflow/models/research/object_detection/core:model",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/protos:train_py_pb2",
    ],
)

py_library(
    name = "eval_util",
    srcs = [
        "eval_util.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:box_list",
        "//tensorflow/models/research/object_detection/core:box_list_ops",
        "//tensorflow/models/research/object_detection/core:keypoint_ops",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:visualization_utils",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection:eval_util",
        "//tensorflow/models/research/object_detection/core:prefetcher",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/protos:eval_py_pb2",
        "//tensorflow/models/research/object_detection/utils:object_detection_evaluation",
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
        "//tensorflow/models/research/object_detection/builders:dataset_builder",
        "//tensorflow/models/research/object_detection/builders:model_builder",
        "//tensorflow/models/research/object_detection/utils:config_util",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
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
        "//tensorflow/models/research/object_detection/builders:model_builder",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/data_decoders:tf_example_decoder",
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
        "//tensorflow/models/research/object_detection/builders:model_builder",
        "//tensorflow/models/research/object_detection/core:model",
        "//tensorflow/models/research/object_detection/protos:pipeline_py_pb2",
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
        "//tensorflow/models/research/object_detection/protos:pipeline_py_pb2",
    ],
)
