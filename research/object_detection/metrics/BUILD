# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_binary(
    name = "offline_eval_map_corloc",
    srcs = [
        "offline_eval_map_corloc.py",
    ],
    deps = [
        ":tf_example_parser",
        "//tensorflow_models/object_detection:evaluator",
        "//tensorflow_models/object_detection/builders:input_reader_builder",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/utils:config_util",
        "//tensorflow_models/object_detection/utils:label_map_util",
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
        "//tensorflow_models/object_detection/core:data_parser",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)

py_test(
    name = "tf_example_parser_test",
    srcs = ["tf_example_parser_test.py"],
    deps = [
        ":tf_example_parser",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)
