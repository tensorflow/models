# TFLite ops for sequence string projection.
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])

package(
    default_visibility = ["//:friends"],  # sequence projection
)

cc_library(
    name = "sequence_string_projection",
    srcs = ["sequence_string_projection.cc"],
    hdrs = ["sequence_string_projection.h"],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "//tf_ops:projection_normalizer_util",  # sequence projection
        "//tf_ops:projection_util",  # sequence projection
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sequence_string_projection_test",
    size = "small",
    srcs = ["sequence_string_projection_test.cc"],
    deps = [
        ":sequence_string_projection",
        ":tf_tflite_diff_test_util",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        # "//tf_ops:sequence_string_projection_op"  # sequence projection
        "//tf_ops:sequence_string_projection_op_v2",  # sequence projection
    ],
)

cc_library(
    name = "tf_tflite_diff_test_util",
    testonly = 1,
    srcs = ["tf_tflite_diff_test_util.cc"],
    hdrs = ["tf_tflite_diff_test_util.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_util",
    hdrs = ["quantization_util.h"],
    deps = ["@org_tensorflow//tensorflow/lite:context"],
)

cc_library(
    name = "expected_value",
    srcs = ["expected_value.cc"],
    hdrs = ["expected_value.h"],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "layer_norm",
    srcs = ["layer_norm.cc"],
    hdrs = ["layer_norm.h"],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_test(
    name = "layer_norm_test",
    size = "small",
    srcs = ["layer_norm_test.cc"],
    deps = [
        ":layer_norm",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
