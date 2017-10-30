# Tensorflow Object Detection API: data decoders.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])
# Apache 2.0

py_library(
    name = "tf_example_decoder",
    srcs = ["tf_example_decoder.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:data_decoder",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/utils:label_map_util",
    ],
)

py_test(
    name = "tf_example_decoder_test",
    srcs = ["tf_example_decoder_test.py"],
    deps = [
        ":tf_example_decoder",
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
    ],
)
