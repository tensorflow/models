# Contains functions related to implementing the LSTM cell. Split out into a
# folder because we will probably add different test harnesses, data, etc.

package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "if_linux_x86_64",
)
load(
    "//dragnn/runtime:multiarch.bzl",
    "dragnn_cc_multiarch_library",
    "dragnn_cc_multiarch_test",
    "dragnn_cc_multiarch_binary",
)

FAST_MATH_COPTS = if_linux_x86_64([
    # Note: Without masking, -O3 is significantly faster.
    "-O3",
    "-msse4.2",
    "-ffast-math",
])

dragnn_cc_multiarch_library(
    name = "cell_function",
    srcs = ["cell_function.cc"],
    hdrs = ["cell_function.h"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        "//dragnn/runtime/math:avx_activation_functions",
        "//dragnn/runtime/math:avx_vector_array",
        "//dragnn/runtime/math:sgemvv",
        "//dragnn/runtime/math:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_library(
    name = "test_helpers",
    testonly = 1,
    hdrs = ["test_helpers.h"],
    deps = [
        ":cell_function",
        "//dragnn/runtime/math:float16_types",
        "//dragnn/runtime/math:sgemvv",
        "//dragnn/runtime/test:helpers",
    ],
)

dragnn_cc_multiarch_test(
    name = "cell_function_test",
    srcs = ["cell_function_test.cc"],
    deps = [
        ":cell_function",
        ":test_helpers",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/math:arithmetic",
        "//dragnn/runtime/math:transformations",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_binary(
    name = "cell_function_benchmark",
    testonly = 1,
    srcs = ["cell_function_benchmark.cc"],
    deps = [
        ":cell_function",
        ":test_helpers",
        "//dragnn/runtime/math:transformations",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
    ],
)
