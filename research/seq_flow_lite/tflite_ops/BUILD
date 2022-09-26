# TFLite ops for sequence string projection.
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts")

licenses(["notice"])

package(
    default_visibility = ["//:friends"],  # sequence projection
)

cc_library(
    name = "beam_search",
    srcs = ["beam_search.cc"],
    hdrs = ["beam_search.h"],
    copts = tflite_copts(),
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "//tflite_ops:quantization_util",  # sequence projection
        "@com_github_google_glog//:glog",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "beam_search_test",
    srcs = ["beam_search_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":beam_search",
        "@org_tensorflow//tensorflow/lite/c:c_api_types",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:legacy_reference_base",
        "@org_tensorflow//tensorflow/lite/kernels/internal:optimized_base",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
        "//tflite_ops:quantization_util",  # sequence projection
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "denylist",
    srcs = ["denylist.cc"],
    hdrs = ["denylist.h"],
    copts = tflite_copts(),
    deps = [
        ":quantization_util",
        "@org_tensorflow//tensorflow/lite:framework",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_library(
    name = "denylist_skipgram",
    srcs = ["denylist_skipgram.cc"],
    hdrs = ["denylist_skipgram.h"],
    copts = tflite_copts(),
    deps = [
        ":denylist",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "//tf_ops:skipgram_finder",  # sequence projection
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_test(
    name = "denylist_skipgram_test",
    size = "small",
    srcs = ["denylist_skipgram_test.cc"],
    deps = [
        ":denylist_skipgram",
        ":tf_tflite_diff_test_util",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "//tf_ops:denylist_op",  # sequence projection
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "denylist_subsequence",
    srcs = ["denylist_subsequence.cc"],
    hdrs = ["denylist_subsequence.h"],
    copts = tflite_copts(),
    deps = [
        ":denylist",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "//tf_ops:subsequence_finder",  # sequence projection
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_test(
    name = "denylist_subsequence_test",
    size = "small",
    srcs = ["denylist_subsequence_test.cc"],
    deps = [
        ":denylist_subsequence",
        ":tf_tflite_diff_test_util",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "//tf_ops:denylist_op",  # sequence projection
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "denylist_tokenized",
    srcs = ["denylist_tokenized.cc"],
    hdrs = ["denylist_tokenized.h"],
    copts = tflite_copts(),
    deps = [
        ":denylist",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "//tf_ops:skipgram_finder",  # sequence projection
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
    ],
)

cc_test(
    name = "denylist_tokenized_test",
    size = "small",
    srcs = ["denylist_tokenized_test.cc"],
    deps = [
        ":denylist_tokenized",
        ":tf_tflite_diff_test_util",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core/kernels:ops_testutil",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "//tf_ops:denylist_op",  # sequence projection
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
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
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
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

cc_library(
    name = "quantization_util",
    hdrs = ["quantization_util.h"],
    deps = ["@org_tensorflow//tensorflow/lite:context"],
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
        "//tf_ops:projection_util",  # sequence projection
        # "//tf_ops:sequence_string_projection_op"  # sequence projection
        # "//tf_ops:sequence_string_projection_op_v2"  # sequence projection
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

pybind_extension(
    name = "registerer",
    srcs = ["registerer.cc"],
    module_name = "registerer",
    deps = [
        ":expected_value",
        ":layer_norm",
        ":sequence_string_projection",
        "@org_tensorflow//tensorflow/lite:framework",
        "@pybind11",
    ],
)

cc_library(
    name = "tflite_qrnn_pooling",
    srcs = ["tflite_qrnn_pooling.cc"],
    hdrs = ["tflite_qrnn_pooling.h"],
    copts = tflite_copts(),
    deps = [
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "//tflite_ops:quantization_util",  # sequence projection
        "@com_google_absl//absl/base:core_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_decoder_cache",
    hdrs = ["tflite_decoder_cache.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tflite_decoder_handler",
    srcs = ["tflite_decoder_handler.cc"],
    hdrs = ["tflite_decoder_handler.h"],
    copts = tflite_copts(),
    deps = [
        ":tflite_decoder_cache",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "//tflite_ops:quantization_util",  # sequence projection
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tflite_decoder_handler_test",
    size = "small",
    srcs = ["tflite_decoder_handler_test.cc"],
    deps = [
        ":tflite_decoder_handler",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)
