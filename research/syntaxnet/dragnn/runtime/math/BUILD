package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "if_linux_x86_64",
)
load(
    "//dragnn/runtime:multiarch.bzl",
    "dragnn_cc_multiarch_test",
)

FAST_MATH_COPTS = if_linux_x86_64([
    "-O3",
    "-msse4.2",
    "-ffast-math",
    "-ftree-vectorize",
])

cc_library(
    name = "avx_vector_array",
    hdrs = ["avx_vector_array.h"],
    deps = [":float16_types"],
)

cc_test(
    name = "avx_vector_array_test",
    srcs = ["avx_vector_array_test.cc"],
    deps = [
        ":avx_vector_array",
        "//dragnn/runtime/test:helpers",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "avx_activation_functions",
    hdrs = ["avx_activation_functions.h"],
    deps = [
        ":avx_vector_array",
    ],
)

dragnn_cc_multiarch_test(
    name = "avx_activation_functions_test",
    srcs = ["avx_activation_functions_test.cc"],
    copts = FAST_MATH_COPTS,
    deps = [
        ":avx_activation_functions",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "float16_types",
    hdrs = ["float16_types.h"],
    deps = [
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "float16_types_test",
    srcs = ["float16_types_test.cc"],
    deps = [
        ":float16_types",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sgemvv",
    hdrs = ["sgemvv.h"],
    deps = [
        ":avx_vector_array",
        ":types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "sgemvv_test",
    srcs = ["sgemvv_test.cc"],
    copts = [
        "-O3",
        "-mavx2",
        "-mfma",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":arithmetic",
        ":sgemvv",
        ":transformations",
        ":types",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/test:helpers",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "sgemvv_compatibility_test",
    srcs = ["sgemvv_test.cc"],
    copts = [
        "-O3",
        "-ftree-vectorize",
        "-ffast-math",
    ],
    deps = [
        ":arithmetic",
        ":sgemvv",
        ":transformations",
        ":types",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/test:helpers",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "transformations",
    hdrs = ["transformations.h"],
    deps = [
        ":types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "transformations_test",
    srcs = ["transformations_test.cc"],
    deps = [
        ":transformations",
        "//dragnn/runtime/test:helpers",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    deps = [
        "//dragnn/runtime:alignment",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        "//dragnn/core/test:generic",
        "//dragnn/runtime:alignment",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "eigen",
    hdrs = ["eigen.h"],
    deps = [
        ":types",
        "//dragnn/runtime:alignment",
        "@org_tensorflow//third_party/eigen3",
    ],
)

cc_test(
    name = "eigen_test",
    size = "small",
    srcs = ["eigen_test.cc"],
    deps = [
        ":eigen",
        ":types",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/test:helpers",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "arithmetic",
    srcs = [
        "arithmetic_avx.h",
        "arithmetic_common.h",
        "arithmetic_neon.h",
        "arithmetic_sse.h",
    ],
    hdrs = ["arithmetic.h"],
    deps = [
        ":types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "arithmetic_test",
    size = "small",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        ":types",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "arithmetic_avx_test",
    size = "small",
    srcs = ["arithmetic_test.cc"],
    copts = [
        "-mavx2",
        "-mfma",
    ],
    tags = [
        "manual",
    ],
    deps = [
        ":arithmetic",
        ":types",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "arithmetic_sse_test",
    size = "small",
    srcs = ["arithmetic_test.cc"],
    copts = ["-msse4.2"],
    deps = [
        ":arithmetic",
        ":types",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
