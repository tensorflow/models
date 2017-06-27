# Tensorflow Object Detection API: Utility functions.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "category_util",
    srcs = ["category_util.py"],
    deps = ["//tensorflow"],
)

py_library(
    name = "dataset_util",
    srcs = ["dataset_util.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "label_map_util",
    srcs = ["label_map_util.py"],
    deps = [
        "//third_party/py/google/protobuf",
        "//tensorflow",
        "//tensorflow_models/object_detection/protos:string_int_label_map_py_pb2",
    ],
)

py_library(
    name = "learning_schedules",
    srcs = ["learning_schedules.py"],
    deps = ["//tensorflow"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = ["//third_party/py/numpy"],
)

py_library(
    name = "np_box_list",
    srcs = ["np_box_list.py"],
    deps = ["//tensorflow"],
)

py_library(
    name = "np_box_list_ops",
    srcs = ["np_box_list_ops.py"],
    deps = [
        ":np_box_list",
        ":np_box_ops",
        "//tensorflow",
    ],
)

py_library(
    name = "np_box_ops",
    srcs = ["np_box_ops.py"],
    deps = ["//tensorflow"],
)

py_library(
    name = "object_detection_evaluation",
    srcs = ["object_detection_evaluation.py"],
    deps = [
        ":metrics",
        ":per_image_evaluation",
        "//tensorflow",
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":static_shape",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:box_list_ops",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)

py_library(
    name = "per_image_evaluation",
    srcs = ["per_image_evaluation.py"],
    deps = [
        ":np_box_list",
        ":np_box_list_ops",
        "//tensorflow",
    ],
)

py_library(
    name = "shape_utils",
    srcs = ["shape_utils.py"],
    deps = ["//tensorflow"],
)

py_library(
    name = "static_shape",
    srcs = ["static_shape.py"],
    deps = [],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:anchor_generator",
        "//tensorflow_models/object_detection/core:box_coder",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:box_predictor",
        "//tensorflow_models/object_detection/core:matcher",
    ],
)

py_library(
    name = "variables_helper",
    srcs = ["variables_helper.py"],
    deps = [
        "//tensorflow",
    ],
)

py_library(
    name = "visualization_utils",
    srcs = ["visualization_utils.py"],
    deps = [
        "//third_party/py/PIL:pil",
        "//tensorflow",
    ],
)

py_test(
    name = "category_util_test",
    srcs = ["category_util_test.py"],
    deps = [
        ":category_util",
        "//tensorflow",
    ],
)

py_test(
    name = "dataset_util_test",
    srcs = ["dataset_util_test.py"],
    deps = [
        ":dataset_util",
        "//tensorflow",
    ],
)

py_test(
    name = "label_map_util_test",
    srcs = ["label_map_util_test.py"],
    deps = [
        ":label_map_util",
        "//tensorflow",
    ],
)

py_test(
    name = "learning_schedules_test",
    srcs = ["learning_schedules_test.py"],
    deps = [
        ":learning_schedules",
        "//tensorflow",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":metrics",
        "//tensorflow",
    ],
)

py_test(
    name = "np_box_list_test",
    srcs = ["np_box_list_test.py"],
    deps = [
        ":np_box_list",
        "//tensorflow",
    ],
)

py_test(
    name = "np_box_list_ops_test",
    srcs = ["np_box_list_ops_test.py"],
    deps = [
        ":np_box_list",
        ":np_box_list_ops",
        "//tensorflow",
    ],
)

py_test(
    name = "np_box_ops_test",
    srcs = ["np_box_ops_test.py"],
    deps = [
        ":np_box_ops",
        "//tensorflow",
    ],
)

py_test(
    name = "object_detection_evaluation_test",
    srcs = ["object_detection_evaluation_test.py"],
    deps = [
        ":object_detection_evaluation",
        "//tensorflow",
    ],
)

py_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    deps = [
        ":ops",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)

py_test(
    name = "per_image_evaluation_test",
    srcs = ["per_image_evaluation_test.py"],
    deps = [
        ":per_image_evaluation",
        "//tensorflow",
    ],
)

py_test(
    name = "shape_utils_test",
    srcs = ["shape_utils_test.py"],
    deps = [
        ":shape_utils",
        "//tensorflow",
    ],
)

py_test(
    name = "static_shape_test",
    srcs = ["static_shape_test.py"],
    deps = [
        ":static_shape",
        "//tensorflow",
    ],
)

py_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    deps = [
        ":test_utils",
        "//tensorflow",
    ],
)

py_test(
    name = "variables_helper_test",
    srcs = ["variables_helper_test.py"],
    deps = [
        ":variables_helper",
        "//tensorflow",
    ],
)

py_test(
    name = "visualization_utils_test",
    srcs = ["visualization_utils_test.py"],
    deps = [
        ":visualization_utils",
        "//third_party/py/PIL:pil",
    ],
)
