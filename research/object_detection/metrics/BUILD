# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "coco_tools",
    srcs = [
        "coco_tools.py",
    ],
    deps = [
        "//file/localfile",
        "//file/placer",
        "//pycocotools",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:json_utils",
    ],
)

py_test(
    name = "coco_tools_test",
    srcs = [
        "coco_tools_test.py",
    ],
    deps = [
        ":coco_tools",
        "//testing/pybase",
        "//numpy",
    ],
)

py_library(
    name = "coco_evaluation",
    srcs = [
        "coco_evaluation.py",
    ],
    deps = [
        ":coco_tools",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/utils:object_detection_evaluation",
    ],
)

py_test(
    name = "coco_evaluation_test",
    srcs = [
        "coco_evaluation_test.py",
    ],
    deps = [
        ":coco_evaluation",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
    ],
)

py_binary(
    name = "offline_eval_map_corloc",
    srcs = [
        "offline_eval_map_corloc.py",
    ],
    deps = [
        ":tf_example_parser",
        "//tensorflow/models/research/object_detection:evaluator",
        "//tensorflow/models/research/object_detection/builders:input_reader_builder",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/utils:config_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
    ],
)

py_test(
    name = "offline_eval_map_corloc_test",
    srcs = [
        "offline_eval_map_corloc_test.py",
    ],
    deps = [
        ":offline_eval_map_corloc",
        "//tensorflow",
    ],
)

py_library(
    name = "tf_example_parser",
    srcs = ["tf_example_parser.py"],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:data_parser",
        "//tensorflow/models/research/object_detection/core:standard_fields",
    ],
)

py_test(
    name = "tf_example_parser_test",
    srcs = ["tf_example_parser_test.py"],
    deps = [
        ":tf_example_parser",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
    ],
)
