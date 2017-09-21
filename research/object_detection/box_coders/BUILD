# Tensorflow Object Detection API: Box Coder implementations.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0
py_library(
    name = "faster_rcnn_box_coder",
    srcs = [
        "faster_rcnn_box_coder.py",
    ],
    deps = [
        "//tensorflow_models/object_detection/core:box_coder",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)

py_test(
    name = "faster_rcnn_box_coder_test",
    srcs = [
        "faster_rcnn_box_coder_test.py",
    ],
    deps = [
        ":faster_rcnn_box_coder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)

py_library(
    name = "keypoint_box_coder",
    srcs = [
        "keypoint_box_coder.py",
    ],
    deps = [
        "//tensorflow_models/object_detection/core:box_coder",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)

py_test(
    name = "keypoint_box_coder_test",
    srcs = [
        "keypoint_box_coder_test.py",
    ],
    deps = [
        ":keypoint_box_coder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)

py_library(
    name = "mean_stddev_box_coder",
    srcs = [
        "mean_stddev_box_coder.py",
    ],
    deps = [
        "//tensorflow_models/object_detection/core:box_coder",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)

py_test(
    name = "mean_stddev_box_coder_test",
    srcs = [
        "mean_stddev_box_coder_test.py",
    ],
    deps = [
        ":mean_stddev_box_coder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)

py_library(
    name = "square_box_coder",
    srcs = [
        "square_box_coder.py",
    ],
    deps = [
        "//tensorflow_models/object_detection/core:box_coder",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)

py_test(
    name = "square_box_coder_test",
    srcs = [
        "square_box_coder_test.py",
    ],
    deps = [
        ":square_box_coder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:box_list",
    ],
)
