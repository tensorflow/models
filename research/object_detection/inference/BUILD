# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "detection_inference",
    srcs = ["detection_inference.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)

py_test(
    name = "detection_inference_test",
    srcs = ["detection_inference_test.py"],
    deps = [
        ":detection_inference",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/utils:dataset_util",
    ],
)

py_binary(
    name = "infer_detections",
    srcs = ["infer_detections.py"],
    deps = [
        ":detection_inference",
        "//tensorflow",
    ],
)
