package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "helpers",
    testonly = 1,
    srcs = ["helpers.cc"],
    hdrs = ["helpers.h"],
    deps = [
        "//dragnn/runtime:alignment",
        "//dragnn/runtime/math:avx_vector_array",
        "//dragnn/runtime/math:sgemvv",
        "//dragnn/runtime/math:transformations",
        "//dragnn/runtime/math:types",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "helpers_test",
    size = "small",
    srcs = ["helpers_test.cc"],
    deps = [
        ":helpers",
        "//dragnn/runtime:alignment",
        "//dragnn/runtime/math:types",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "fake_variable_store",
    testonly = 1,
    srcs = ["fake_variable_store.cc"],
    hdrs = ["fake_variable_store.h"],
    deps = [
        ":helpers",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime:alignment",
        "//dragnn/runtime:variable_store",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "fake_variable_store_test",
    size = "small",
    srcs = ["fake_variable_store_test.cc"],
    deps = [
        ":fake_variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/runtime:alignment",
        "//dragnn/runtime/math:types",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "network_test_base",
    testonly = 1,
    srcs = ["network_test_base.cc"],
    hdrs = ["network_test_base.h"],
    deps = [
        ":fake_variable_store",
        "//dragnn/core/test:mock_compute_session",
        "//dragnn/protos:data_proto_cc",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:flexible_matrix_kernel",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "term_map_helpers",
    testonly = 1,
    srcs = ["term_map_helpers.cc"],
    hdrs = ["term_map_helpers.h"],
    deps = [
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "term_map_helpers_test",
    size = "small",
    srcs = ["term_map_helpers_test.cc"],
    deps = [
        ":term_map_helpers",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:term_frequency_map",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
