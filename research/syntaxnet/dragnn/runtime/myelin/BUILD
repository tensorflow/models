package(default_visibility = ["//visibility:public"])

load(
    ":build_defs.bzl",
    "dragnn_myelin_cc_library",
    "dragnn_myelin_cc_test",
    "dragnn_myelin_cc_multiarch_library",
    "dragnn_myelin_cc_multiarch_test",
)

test_suite(name = "all_tests")

filegroup(
    name = "test_myelination_output",
    srcs = glob(["testdata/myelination_output/**"]),
)

cc_library(
    name = "attr_value_utils",
    srcs = ["attr_value_utils.cc"],
    hdrs = ["attr_value_utils.h"],
    deps = [
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "attr_value_utils_test",
    size = "small",
    srcs = ["attr_value_utils_test.cc"],
    deps = [
        ":attr_value_utils",
        "//dragnn/core/test:generic",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_myelin_cc_library(
    name = "myelin_cell_converter",
    srcs = ["myelin_cell_converter.cc"],
    hdrs = ["myelin_cell_converter.h"],
    deps = [
        ":attr_value_utils",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/runtime:trained_model",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

dragnn_myelin_cc_test(
    name = "myelin_cell_converter_test",
    size = "small",
    timeout = "moderate",
    srcs = ["myelin_cell_converter_test.cc"],
    data = ["//dragnn/runtime:test_rnn_tagger"],
    deps = [
        ":myelin_cell_converter",
        ":myelin_spec_utils",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//dragnn/runtime:alignment",
        "//dragnn/runtime:trained_model",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@sling//sling/myelin:compute",
        "@sling//sling/myelin:flow",
        "@sling//sling/myelin:graph",
    ],
)

dragnn_myelin_cc_library(
    name = "myelin_library",
    srcs = ["myelin_library.cc"],
    hdrs = ["myelin_library.h"],
    deps = [
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/myelin:flow",
    ],
)

dragnn_myelin_cc_test(
    name = "myelin_library_test",
    size = "small",
    srcs = ["myelin_library_test.cc"],
    deps = [
        ":myelin_library",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_myelin_cc_library(
    name = "myelin_spec_utils",
    srcs = ["myelin_spec_utils.cc"],
    hdrs = ["myelin_spec_utils.h"],
    deps = [
        ":myelin_library",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/base",
        "@sling//sling/file",
        "@sling//sling/myelin:compute",
        "@sling//sling/myelin:flow",
        "@sling//sling/myelin/kernel:tensorflow",
    ],
)

dragnn_myelin_cc_test(
    name = "myelin_spec_utils_test",
    size = "small",
    srcs = ["myelin_spec_utils_test.cc"],
    deps = [
        ":myelin_spec_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@sling//sling/file",
        "@sling//sling/file:posix",
        "@sling//sling/myelin:compute",
        "@sling//sling/myelin:flow",
    ],
)

dragnn_myelin_cc_library(
    name = "myelin_tracing",
    srcs = ["myelin_tracing.cc"],
    hdrs = ["myelin_tracing.h"],
    deps = [
        "//dragnn/protos:cell_trace_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/myelin:compute",
    ],
)

dragnn_myelin_cc_test(
    name = "myelin_tracing_test",
    size = "small",
    srcs = ["myelin_tracing_test.cc"],
    deps = [
        ":myelin_spec_utils",
        ":myelin_tracing",
        "//dragnn/core/test:generic",
        "//dragnn/protos:cell_trace_proto_cc",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@sling//sling/myelin:compute",
        "@sling//sling/myelin:flow",
    ],
)

dragnn_myelin_cc_multiarch_library(
    name = "myelin_dynamic_component_base",
    srcs = ["myelin_dynamic_component_base.cc"],
    hdrs = ["myelin_dynamic_component_base.h"],
    deps = [
        ":myelin_spec_utils",
        ":myelin_tracing",
        "//dragnn/protos:cell_trace_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:alignment",
        "//dragnn/runtime:component",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:fixed_embeddings",
        "//dragnn/runtime:linked_embeddings",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime:transition_system_traits",
        "//dragnn/runtime:variable_store",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/myelin:compute",
        "@sling//sling/myelin:flow",
    ],
)

dragnn_myelin_cc_multiarch_library(
    name = "myelin_dynamic_component",
    srcs = ["myelin_dynamic_component.cc"],
    deps = [
        ":myelin_dynamic_component_base",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:fixed_embeddings",
        "//dragnn/runtime:linked_embeddings",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/myelin:compute",
    ],
    alwayslink = 1,
)

dragnn_myelin_cc_multiarch_test(
    name = "myelin_dynamic_component_test",
    size = "small",
    srcs = ["myelin_dynamic_component_test.cc"],
    deps = [
        ":myelin_dynamic_component",
        ":myelin_spec_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:cell_trace_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@sling//sling/file",
        "@sling//sling/file:posix",
        "@sling//sling/myelin:flow",
    ],
)

dragnn_myelin_cc_library(
    name = "myelination",
    srcs = ["myelination.cc"],
    hdrs = ["myelination.h"],
    deps = [
        ":myelin_cell_converter",
        ":myelin_spec_utils",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:trained_model",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_myelin_cc_test(
    name = "myelination_test",
    size = "small",
    timeout = "moderate",
    srcs = ["myelination_test.cc"],
    data = [
        ":test_myelination_output",
        "//dragnn/runtime:test_rnn_tagger",
    ],
    deps = [
        ":myelin_spec_utils",
        ":myelination",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_myelin_cc_multiarch_library(
    name = "sequence_myelin_dynamic_component",
    srcs = ["sequence_myelin_dynamic_component.cc"],
    deps = [
        ":myelin_dynamic_component_base",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:sequence_features",
        "//dragnn/runtime:sequence_links",
        "//dragnn/runtime:sequence_model",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime:variable_store",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/myelin:compute",
    ],
    alwayslink = 1,
)

dragnn_myelin_cc_multiarch_test(
    name = "sequence_myelin_dynamic_component_test",
    size = "small",
    srcs = ["sequence_myelin_dynamic_component_test.cc"],
    deps = [
        ":myelin_spec_utils",
        ":sequence_myelin_dynamic_component",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:sequence_backend",
        "//dragnn/runtime:sequence_extractor",
        "//dragnn/runtime:sequence_linker",
        "//dragnn/runtime:sequence_predictor",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime:variable_store",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@sling//sling/file",
        "@sling//sling/file:posix",
        "@sling//sling/myelin:flow",
    ],
)
