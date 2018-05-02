package(default_visibility = ["//visibility:public"])

# TODO(googleuser): Move XLA libs to dragnn/runtime when stable. Probably there
# should be a refactor with the Myelin libs since they are so similar.

load(
    "//dragnn/runtime/xla:xla_build_defs.bzl",
    "dragnn_xla_aot_components",
)
load(
    "//dragnn/runtime:multiarch.bzl",
    "dragnn_cc_multiarch_library",
    "dragnn_cc_multiarch_test",
)

filegroup(
    name = "test_xla_compilation_output",
    srcs = glob(["testdata/xla_compilation_output/**"]),
)

cc_binary(
    name = "xla_extract_config",
    srcs = ["xla_extract_config.cc"],
    deps = [
        ":xla_graph_utils",
        "//dragnn/protos:export_proto_cc",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_binary(
    name = "xla_extract_names_from_specs",
    srcs = ["xla_extract_names_from_specs.cc"],
    deps = [
        ":xla_spec_build_utils",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "xla_cell_converter",
    srcs = ["xla_cell_converter.cc"],
    hdrs = ["xla_cell_converter.h"],
    deps = [
        ":xla_graph_utils",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/runtime:trained_model",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "xla_cell_converter_test",
    size = "small",
    timeout = "moderate",
    srcs = ["xla_cell_converter_test.cc"],
    data = ["//dragnn/runtime:test_rnn_tagger"],
    deps = [
        ":xla_cell_converter",
        ":xla_graph_utils",
        ":xla_spec_utils",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/runtime:alignment",
        "//dragnn/runtime:trained_model",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto",
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_jit_compiled_cpu_function",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "xla_compilation",
    srcs = ["xla_compilation.cc"],
    hdrs = ["xla_compilation.h"],
    deps = [
        ":xla_cell_converter",
        ":xla_graph_utils",
        ":xla_spec_utils",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:trained_model",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "xla_compilation_test",
    size = "small",
    timeout = "moderate",
    srcs = ["xla_compilation_test.cc"],
    data = [
        ":test_xla_compilation_output",
        "//dragnn/runtime:test_rnn_tagger",
    ],
    deps = [
        ":xla_compilation",
        ":xla_spec_utils",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "xla_dynamic_component_base",
    srcs = ["xla_dynamic_component_base.cc"],
    hdrs = ["xla_dynamic_component_base.h"],
    deps = [
        ":xla_spec_utils",
        "//dragnn/protos:export_proto_cc",
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
        "//dragnn/runtime:type_keyed_set",
        "//dragnn/runtime:variable_store",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

dragnn_cc_multiarch_library(
    name = "sequence_xla_dynamic_component_mixin",
    hdrs = ["sequence_xla_dynamic_component_mixin.h"],
    deps = [
        ":xla_dynamic_component_base",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:sequence_features",
        "//dragnn/runtime:sequence_links",
        "//dragnn/runtime:sequence_model",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime:variable_store",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "sequence_xla_dynamic_component_mixin_test",
    size = "small",
    srcs = ["sequence_xla_dynamic_component_mixin_test.cc"],
    deps = [
        ":xla_dynamic_component",
        ":xla_graph_utils",
        ":xla_spec_utils",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:cell_trace_proto_cc",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:sequence_backend",
        "//dragnn/runtime:sequence_extractor",
        "//dragnn/runtime:sequence_predictor",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "xla_aot_dynamic_component",
    hdrs = ["xla_aot_dynamic_component.h"],
    deps = [
        ":sequence_xla_dynamic_component_mixin",
        ":xla_dynamic_component_base",
        ":xla_graph_utils",
        ":xla_spec_utils",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime:component",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto",
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

dragnn_cc_multiarch_library(
    name = "xla_dynamic_component",
    srcs = ["xla_dynamic_component.cc"],
    deps = [
        ":sequence_xla_dynamic_component_mixin",
        ":xla_dynamic_component_base",
        ":xla_graph_utils",
        ":xla_spec_utils",
        "//dragnn/core:compute_session",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:fixed_embeddings",
        "//dragnn/runtime:linked_embeddings",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto",
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_compiled_cpu_function",
        "@org_tensorflow//tensorflow/compiler/tf2xla:xla_jit_compiled_cpu_function",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "xla_dynamic_component_test",
    size = "small",
    srcs = ["xla_dynamic_component_test.cc"],
    deps = [
        ":xla_dynamic_component",
        ":xla_graph_utils",
        ":xla_spec_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:cell_trace_proto_cc",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime:component",
        "//dragnn/runtime:extensions",
        "//dragnn/runtime:network_states",
        "//dragnn/runtime:session_state",
        "//dragnn/runtime:type_keyed_set",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:fake_variable_store",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "xla_graph_utils",
    srcs = ["xla_graph_utils.cc"],
    hdrs = ["xla_graph_utils.h"],
    deps = [
        ":xla_spec_utils",
        "//dragnn/protos:export_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "xla_graph_utils_test",
    srcs = ["xla_graph_utils_test.cc"],
    deps = [
        ":xla_graph_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:export_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/compiler/tf2xla:tf2xla_proto",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "xla_spec_build_utils",
    srcs = ["xla_spec_build_utils.cc"],
    hdrs = ["xla_spec_build_utils.h"],
    deps = [
        ":xla_spec_utils",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "xla_spec_build_utils_test",
    srcs = ["xla_spec_build_utils_test.cc"],
    deps = [
        ":xla_spec_build_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "xla_spec_utils",
    srcs = ["xla_spec_utils.cc"],
    hdrs = ["xla_spec_utils.h"],
    deps = [
        "//dragnn/protos:export_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "xla_spec_utils_test",
    srcs = ["xla_spec_utils_test.cc"],
    deps = [
        ":xla_spec_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
