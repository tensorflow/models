# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

load("//learning/brain/contrib/learn/tpu:tpu.bzl", "cloud_tpu_py_binaries")

licenses(["notice"])

# Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "inputs",
    srcs = [
        "inputs.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/builders:dataset_builder",
        "//tensorflow/models/research/object_detection/builders:image_resizer_builder",
        "//tensorflow/models/research/object_detection/builders:model_builder",
        "//tensorflow/models/research/object_detection/builders:preprocessor_builder",
        "//tensorflow/models/research/object_detection/protos:input_reader_py_pb2",
        "//tensorflow/models/research/object_detection/protos:model_py_pb2",
        "//tensorflow/models/research/object_detection/protos:train_py_pb2",
        "//tensorflow/models/research/object_detection/utils:config_util",
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
    name = "model",
    srcs = [
        "model.py",
    ],
    deps = [
        ":inputs",
        ":model_hparams",
        "//tensorflow",
        "//tensorflow/models/research/object_detection:eval_util",
        "//tensorflow/models/research/object_detection/builders:model_builder",
        "//tensorflow/models/research/object_detection/builders:optimizer_builder",
        "//tensorflow/models/research/object_detection/metrics:coco_evaluation",
        "//tensorflow/models/research/object_detection/utils:config_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:shape_utils",
        "//tensorflow/models/research/object_detection/utils:variables_helper",
        "//tensorflow/models/research/object_detection/utils:visualization_utils",
    ],
)

py_library(
    name = "model_hparams",
    srcs = [
        "model_hparams.py",
    ],
    deps = [
        "//tensorflow",
    ],
)

py_test(
    name = "model_test",
    timeout = "long",
    srcs = [
        "model_test.py",
    ],
    data = [
        "//tensorflow/models/research/object_detection/data:pet_label_map.pbtxt",
        "//tensorflow/models/research/object_detection/samples/configs:faster_rcnn_resnet50_pets.config",
        "//tensorflow/models/research/object_detection/samples/configs:ssd_inception_v2_pets.config",
        "//tensorflow/models/research/object_detection/test_data:pets_examples.record",
    ],
    deps = [
        ":inputs",
        ":model",
        ":model_hparams",
        ":model_test_util",
        "//mock",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/data_decoders:tf_example_decoder",
        "//tensorflow/models/research/object_detection/utils:config_util",
        "//tensorflow/models/research/object_detection/utils:ops",
    ],
)

MODEL_TPU_DEPS = [
    ":inputs",
    ":model",
    ":model_hparams",
    "//tensorflow",
    "//tensorflow/models/research/object_detection:eval_util",
    "//tensorflow/models/research/object_detection/builders:model_builder",
    "//tensorflow/models/research/object_detection/builders:optimizer_builder",
    "//tensorflow/models/research/object_detection/metrics:coco_evaluation",
    "//tensorflow/models/research/object_detection/utils:config_util",
    "//tensorflow/models/research/object_detection/utils:label_map_util",
    "//tensorflow/models/research/object_detection/utils:ops",
    "//tensorflow/models/research/object_detection/utils:variables_helper",
    "//tensorflow/models/research/object_detection/utils:visualization_utils",
]

cloud_tpu_py_binaries(
    name = "model_tpu",
    srcs = [
        "model_tpu.py",
    ],
    main = "model_tpu.py",
    deps = MODEL_TPU_DEPS,
)

py_library(
    name = "model_tpu_lib",
    srcs = [
        "model_tpu.py",
    ],
    deps = MODEL_TPU_DEPS,
)

py_library(
    name = "model_test_util",
    srcs = [
        "model_test_util.py",
    ],
    deps = [
        ":model",
        ":model_hparams",
        "//tensorflow",
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
        "//tensorflow/models/research/object_detection/metrics:coco_evaluation",
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
