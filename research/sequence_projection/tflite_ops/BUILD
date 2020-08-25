# TFLite ops for sequence string projection.
load("//third_party/tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])

package(
    default_visibility = [
        "//sequence_projection:__subpackages__",
    ],
)

cc_library(
    name = "sequence_string_projection",
    srcs = ["sequence_string_projection.cc"],
    hdrs = ["sequence_string_projection.h"],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        "@com_github_google_flatbuffers//:flatbuffers",
        "//third_party/tensorflow/lite:string_util",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
        "//sequence_projection/tf_ops:projection_normalizer_util",
        "//sequence_projection/tf_ops:projection_util",
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
        "@com_github_google_flatbuffers//:flatbuffers",
        "//third_party/tensorflow/lite:framework",
        "//third_party/tensorflow/lite:string_util",
        "//third_party/tensorflow/lite/core/api",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
        "//third_party/tensorflow/lite/kernels:test_util",
        "//sequence_projection/tf_ops:sequence_string_projection_op",
        "//sequence_projection/tf_ops:sequence_string_projection_op_v2",
    ],
)

cc_library(
    name = "tf_tflite_diff_test_util",
    testonly = 1,
    srcs = ["tf_tflite_diff_test_util.cc"],
    hdrs = ["tf_tflite_diff_test_util.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_github_google_flatbuffers//:flatbuffers",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core:test",
        "//third_party/tensorflow/core/kernels:ops_testutil",
        "//third_party/tensorflow/lite/kernels:test_util",
    ],
)

cc_library(
    name = "quantization_util",
    hdrs = ["quantization_util.h"],
    deps = ["//third_party/tensorflow/lite:context"],
)

cc_library(
    name = "expected_value",
    srcs = ["expected_value.cc"],
    hdrs = ["expected_value.h"],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        "//third_party/tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)
