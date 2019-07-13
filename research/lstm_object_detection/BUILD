package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    python_version = "PY2",
    deps = [
        ":trainer",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow_models/lstm_object_detection:model_builder",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/inputs:seq_dataset_builder",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/utils:config_util",
        "@org_tensorflow//tensorflow_models/object_detection/builders:dataset_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:preprocessor_builder",
        "@org_tensorflow//tensorflow_models/object_detection/legacy:trainer",
        "@org_tensorflow//tensorflow_models/object_detection/utils:dataset_util",
    ],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow_models/object_detection/builders:optimizer_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:preprocessor_builder",
        "@org_tensorflow//tensorflow_models/object_detection/core:batcher",
        "@org_tensorflow//tensorflow_models/object_detection/core:preprocessor",
        "@org_tensorflow//tensorflow_models/object_detection/core:standard_fields",
        "@org_tensorflow//tensorflow_models/object_detection/utils:ops",
        "@org_tensorflow//tensorflow_models/object_detection/utils:variables_helper",
        "@org_tensorflow//tensorflow_models/slim:model_deploy",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    deps = [
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/metrics:coco_evaluation_all_frames",
        "@org_tensorflow//tensorflow_models/object_detection:eval_util",
        "@org_tensorflow//tensorflow_models/object_detection/core:prefetcher",
        "@org_tensorflow//tensorflow_models/object_detection/core:standard_fields",
        "@org_tensorflow//tensorflow_models/object_detection/metrics:coco_evaluation",
        "@org_tensorflow//tensorflow_models/object_detection/protos:eval_py_pb2",
        "@org_tensorflow//tensorflow_models/object_detection/utils:object_detection_evaluation",
    ],
)

py_binary(
    name = "eval",
    srcs = [
        "eval.py",
    ],
    python_version = "PY2",
    deps = [
        ":evaluator",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow_models/lstm_object_detection:model_builder",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/inputs:seq_dataset_builder",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/utils:config_util",
        "@org_tensorflow//tensorflow_models/object_detection/builders:dataset_builder",
        "@org_tensorflow//tensorflow_models/object_detection/legacy:evaluator",
        "@org_tensorflow//tensorflow_models/object_detection/utils:config_util",
        "@org_tensorflow//tensorflow_models/object_detection/utils:dataset_util",
        "@org_tensorflow//tensorflow_models/object_detection/utils:label_map_util",
    ],
)

py_library(
    name = "model_builder",
    srcs = ["model_builder.py"],
    deps = [
        "@org_tensorflow//tensorflow_models/lstm_object_detection/meta_architectures:lstm_ssd_meta_arch",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/models:lstm_ssd_interleaved_mobilenet_v2_feature_extractor",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/models:lstm_ssd_mobilenet_v1_feature_extractor",
        "@org_tensorflow//tensorflow_models/object_detection/builders:anchor_generator_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:box_coder_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:box_predictor_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:hyperparams_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:image_resizer_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:losses_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:matcher_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:model_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:post_processing_builder",
        "@org_tensorflow//tensorflow_models/object_detection/builders:region_similarity_calculator_builder",
        "@org_tensorflow//tensorflow_models/object_detection/protos:losses_py_pb2",
        "@org_tensorflow//tensorflow_models/object_detection/protos:model_py_pb2",
    ],
)

py_test(
    name = "model_builder_test",
    srcs = ["model_builder_test.py"],
    python_version = "PY2",
    deps = [
        ":model_builder",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/meta_architectures:lstm_ssd_meta_arch",
        "@org_tensorflow//tensorflow_models/lstm_object_detection/protos:pipeline_py_pb2",
        "@org_tensorflow//tensorflow_models/object_detection/protos:model_py_pb2",
    ],
)
