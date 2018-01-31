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
        "//tensorflow/models/research/object_detection/core:data_decoder",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/protos:input_reader_py_pb2",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
    ],
)

py_test(
    name = "tf_example_decoder_test",
    srcs = ["tf_example_decoder_test.py"],
    deps = [
        ":tf_example_decoder",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/protos:input_reader_py_pb2",
    ],
)
