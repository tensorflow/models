package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "if_linux_x86_64",
)
load(
    "//dragnn/runtime:multiarch.bzl",
    "dragnn_cc_multiarch_binary",
    "dragnn_cc_multiarch_library",
    "dragnn_cc_multiarch_test",
)

FAST_MATH_COPTS = if_linux_x86_64([
    "-O3",
    "-msse4.2",
    "-ffast-math",
    "-ftree-vectorize",
])

filegroup(
    name = "test_rnn_tagger",
    srcs = glob(["testdata/rnn_tagger/**"]),
)

cc_library(
    name = "alignment",
    hdrs = ["alignment.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "alignment_test",
    size = "small",
    srcs = ["alignment_test.cc"],
    deps = [
        ":alignment",
        "//dragnn/core/test:generic",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "mmap",
    srcs = ["mmap.cc"],
    hdrs = ["mmap.h"],
    deps = [
        ":alignment",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "mmap_test",
    size = "small",
    srcs = ["mmap_test.cc"],
    data = [
        "testdata/empty_file",
        "testdata/ten_bytes",
    ],
    deps = [
        ":mmap",
        "//dragnn/core/test:generic",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "operands",
    srcs = ["operands.cc"],
    hdrs = ["operands.h"],
    deps = [
        ":alignment",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "operands_test",
    size = "small",
    srcs = ["operands_test.cc"],
    deps = [
        ":alignment",
        ":operands",
        "//dragnn/runtime/math:types",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "variable_store",
    hdrs = ["variable_store.h"],
    deps = [
        ":alignment",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "variable_store_test",
    size = "small",
    srcs = ["variable_store_test.cc"],
    deps = [
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/test:fake_variable_store",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "trained_model",
    srcs = ["trained_model.cc"],
    hdrs = ["trained_model.h"],
    deps = [
        "//dragnn/core:dragnn_bulk_ops_cc",
        "//dragnn/core:dragnn_ops_cc",
        "//syntaxnet:base",
        "//syntaxnet:parser_ops_cc",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "trained_model_test",
    size = "small",
    timeout = "moderate",
    srcs = ["trained_model_test.cc"],
    data = [":test_rnn_tagger"],
    deps = [
        ":trained_model",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "trained_model_variable_store",
    srcs = ["trained_model_variable_store.cc"],
    hdrs = ["trained_model_variable_store.h"],
    deps = [
        ":alignment",
        ":trained_model",
        ":variable_store",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "trained_model_variable_store_test",
    size = "small",
    timeout = "moderate",
    srcs = ["trained_model_variable_store_test.cc"],
    data = [":test_rnn_tagger"],
    shard_count = 13,
    deps = [
        ":trained_model_variable_store",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/math:avx_vector_array",
        "//dragnn/runtime/math:float16_types",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "variable_store_wrappers",
    srcs = ["variable_store_wrappers.cc"],
    hdrs = ["variable_store_wrappers.h"],
    deps = [
        ":alignment",
        ":flexible_matrix_kernel",
        ":variable_store",
        "//dragnn/protos:runtime_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "variable_store_wrappers_test",
    size = "small",
    srcs = ["variable_store_wrappers_test.cc"],
    deps = [
        ":flexible_matrix_kernel",
        ":variable_store_wrappers",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/math:transformations",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:fake_variable_store",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "array_variable_store",
    srcs = ["array_variable_store.cc"],
    hdrs = ["array_variable_store.h"],
    deps = [
        ":alignment",
        ":variable_store",
        "//dragnn/protos:runtime_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "array_variable_store_test",
    size = "small",
    srcs = ["array_variable_store_test.cc"],
    data = [
        "testdata/array_variable_store_data",
        "testdata/array_variable_store_spec",
        "testdata/empty_file",
    ],
    deps = [
        ":alignment",
        ":array_variable_store",
        ":file_array_variable_store",
        ":mmap_array_variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "array_variable_store_builder",
    srcs = ["array_variable_store_builder.cc"],
    hdrs = ["array_variable_store_builder.h"],
    deps = [
        ":alignment",
        ":array_variable_store",
        ":variable_store_wrappers",
        "//dragnn/protos:runtime_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "array_variable_store_builder_test",
    size = "small",
    srcs = ["array_variable_store_builder_test.cc"],
    data = [
        "testdata/array_variable_store_data",
        "testdata/array_variable_store_spec",
    ],
    deps = [
        ":alignment",
        ":array_variable_store_builder",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

# Tested in array_variable_store_test.
cc_library(
    name = "file_array_variable_store",
    srcs = ["file_array_variable_store.cc"],
    hdrs = ["file_array_variable_store.h"],
    deps = [
        ":alignment",
        ":array_variable_store",
        "//dragnn/protos:runtime_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

# Tested in array_variable_store_test.
cc_library(
    name = "mmap_array_variable_store",
    srcs = ["mmap_array_variable_store.cc"],
    hdrs = ["mmap_array_variable_store.h"],
    deps = [
        ":array_variable_store",
        ":mmap",
        "//dragnn/protos:runtime_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "network_states",
    srcs = ["network_states.cc"],
    hdrs = ["network_states.h"],
    deps = [
        ":alignment",
        ":operands",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "network_states_test",
    size = "small",
    srcs = ["network_states_test.cc"],
    deps = [
        ":alignment",
        ":network_states",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "extensions",
    srcs = ["extensions.cc"],
    hdrs = ["extensions.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "extensions_test",
    size = "small",
    srcs = ["extensions_test.cc"],
    deps = [
        ":extensions",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "linked_embeddings",
    srcs = ["linked_embeddings.cc"],
    hdrs = ["linked_embeddings.h"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":alignment",
        ":flexible_matrix_kernel",
        ":network_states",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:data_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:arithmetic",
        "//dragnn/runtime/math:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "linked_embeddings_test",
    size = "small",
    srcs = ["linked_embeddings_test.cc"],
    deps = [
        ":linked_embeddings",
        ":network_states",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "fixed_embeddings",
    srcs = ["fixed_embeddings.cc"],
    hdrs = ["fixed_embeddings.h"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":alignment",
        ":network_states",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:arithmetic",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "fixed_embeddings_test",
    size = "small",
    srcs = ["fixed_embeddings_test.cc"],
    deps = [
        ":fixed_embeddings",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "type_keyed_set",
    hdrs = ["type_keyed_set.h"],
)

cc_test(
    name = "type_keyed_set_test",
    size = "small",
    srcs = ["type_keyed_set_test.cc"],
    deps = [
        ":type_keyed_set",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "session_state",
    hdrs = ["session_state.h"],
    deps = [
        ":extensions",
        ":network_states",
    ],
)

cc_library(
    name = "session_state_pool",
    srcs = ["session_state_pool.cc"],
    hdrs = ["session_state_pool.h"],
    deps = [
        ":session_state",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "session_state_pool_test",
    size = "small",
    srcs = ["session_state_pool_test.cc"],
    deps = [
        ":session_state",
        ":session_state_pool",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "bulk_dynamic_component",
    srcs = ["bulk_dynamic_component.cc"],
    deps = [
        ":bulk_network_unit",
        ":component",
        ":extensions",
        ":network_states",
        ":network_unit_base",
        ":transition_system_traits",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "bulk_dynamic_component_test",
    srcs = ["bulk_dynamic_component_test.cc"],
    deps = [
        ":bulk_dynamic_component",
        ":bulk_network_unit",
        ":component",
        ":extensions",
        ":network_states",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "sequence_bulk_dynamic_component",
    srcs = ["sequence_bulk_dynamic_component.cc"],
    deps = [
        ":bulk_network_unit",
        ":component",
        ":extensions",
        ":fixed_embeddings",
        ":linked_embeddings",
        ":network_states",
        ":sequence_model",
        ":session_state",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "sequence_bulk_dynamic_component_test",
    srcs = ["sequence_bulk_dynamic_component_test.cc"],
    deps = [
        ":bulk_network_unit",
        ":component",
        ":extensions",
        ":network_states",
        ":sequence_backend",
        ":sequence_bulk_dynamic_component",
        ":sequence_extractor",
        ":sequence_linker",
        ":sequence_predictor",
        ":variable_store",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "component",
    srcs = ["component.cc"],
    hdrs = ["component.h"],
    deps = [
        ":extensions",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "component_test",
    size = "small",
    srcs = ["component_test.cc"],
    deps = [
        ":component",
        ":extensions",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "lstm_network_kernel",
    srcs = ["lstm_network_kernel.cc"],
    hdrs = ["lstm_network_kernel.h"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":attributes",
        ":extensions",
        ":feed_forward_network_layer",
        ":flexible_matrix_kernel",
        ":network_states",
        ":session_state",
        ":transition_system_traits",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/lstm_cell:cell_function",
        "//dragnn/runtime/math:avx_activation_functions",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "lstm_network_kernel_test",
    srcs = ["lstm_network_kernel_test.cc"],
    deps = [
        ":lstm_network_kernel",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/lstm_cell:cell_function",
        "//dragnn/runtime/test:helpers",
        "//dragnn/runtime/test:network_test_base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "lstm_network",
    srcs = ["lstm_network.cc"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":extensions",
        ":lstm_network_kernel",
        ":network_unit",
        ":network_unit_base",
        ":transition_system_traits",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/lstm_cell:cell_function",
        "//dragnn/runtime/math:avx_activation_functions",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "lstm_network_test",
    srcs = ["lstm_network_test.cc"],
    deps = [
        ":flexible_matrix_kernel",
        ":lstm_network",
        ":network_unit",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/lstm_cell:cell_function",
        "//dragnn/runtime/test:network_test_base",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "bulk_lstm_network",
    srcs = ["bulk_lstm_network.cc"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":bulk_network_unit",
        ":extensions",
        ":lstm_network_kernel",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "bulk_lstm_network_test",
    srcs = ["bulk_lstm_network_test.cc"],
    deps = [
        ":bulk_lstm_network",
        ":bulk_network_unit",
        ":flexible_matrix_kernel",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/lstm_cell:cell_function",
        "//dragnn/runtime/test:helpers",
        "//dragnn/runtime/test:network_test_base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "master",
    srcs = ["master.cc"],
    hdrs = ["master.h"],
    deps = [
        ":component",
        ":extensions",
        ":network_states",
        ":session_state",
        ":session_state_pool",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "master_test",
    size = "small",
    srcs = ["master_test.cc"],
    deps = [
        ":alignment",
        ":component",
        ":extensions",
        ":master",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/core/test:mock_compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime/test:fake_variable_store",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "network_unit",
    srcs = ["network_unit.cc"],
    hdrs = ["network_unit.h"],
    deps = [
        ":extensions",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "network_unit_test",
    size = "small",
    srcs = ["network_unit_test.cc"],
    deps = [
        ":extensions",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "bulk_network_unit",
    srcs = ["bulk_network_unit.cc"],
    hdrs = ["bulk_network_unit.h"],
    deps = [
        ":extensions",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "bulk_network_unit_test",
    size = "small",
    srcs = ["bulk_network_unit_test.cc"],
    deps = [
        ":bulk_network_unit",
        ":extensions",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "dynamic_component",
    srcs = ["dynamic_component.cc"],
    deps = [
        ":component",
        ":extensions",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":transition_system_traits",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "dynamic_component_test",
    size = "small",
    srcs = ["dynamic_component_test.cc"],
    deps = [
        ":component",
        ":dynamic_component",
        ":extensions",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "network_unit_base",
    srcs = ["network_unit_base.cc"],
    hdrs = ["network_unit_base.h"],
    deps = [
        ":extensions",
        ":fixed_embeddings",
        ":linked_embeddings",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "network_unit_base_test",
    size = "small",
    srcs = ["network_unit_base_test.cc"],
    deps = [
        ":extensions",
        ":fixed_embeddings",
        ":linked_embeddings",
        ":network_states",
        ":network_unit_base",
        ":session_state",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "attributes",
    srcs = ["attributes.cc"],
    hdrs = ["attributes.h"],
    deps = [
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "attributes_test",
    size = "small",
    srcs = ["attributes_test.cc"],
    deps = [
        ":attributes",
        "//dragnn/core/test:generic",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "activation_functions",
    hdrs = ["activation_functions.h"],
    deps = [
        "//dragnn/runtime/math:arithmetic",
        "//dragnn/runtime/math:types",
    ],
)

cc_test(
    name = "activation_functions_test",
    size = "small",
    srcs = ["activation_functions_test.cc"],
    deps = [
        ":activation_functions",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:helpers",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "flexible_matrix_kernel",
    srcs = ["flexible_matrix_kernel.cc"],
    hdrs = ["flexible_matrix_kernel.h"],
    deps = [
        ":alignment",
        ":variable_store",
        "//dragnn/runtime/math:arithmetic",
        "//dragnn/runtime/math:avx_vector_array",
        "//dragnn/runtime/math:sgemvv",
        "//dragnn/runtime/math:types",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "flexible_matrix_kernel_test",
    srcs = ["flexible_matrix_kernel_test.cc"],
    copts = FAST_MATH_COPTS,
    deps = [
        ":flexible_matrix_kernel",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/runtime/math:transformations",
        "//dragnn/runtime/test:fake_variable_store",
        "//dragnn/runtime/test:helpers",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "feed_forward_network_layer",
    srcs = ["feed_forward_network_layer.cc"],
    hdrs = ["feed_forward_network_layer.h"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":activation_functions",
        ":flexible_matrix_kernel",
        ":network_states",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "feed_forward_network_layer_test",
    size = "small",
    srcs = ["feed_forward_network_layer_test.cc"],
    deps = [
        ":activation_functions",
        ":feed_forward_network_layer",
        ":flexible_matrix_kernel",
        ":network_states",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:helpers",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "feed_forward_network_kernel",
    srcs = ["feed_forward_network_kernel.cc"],
    hdrs = ["feed_forward_network_kernel.h"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":activation_functions",
        ":attributes",
        ":feed_forward_network_layer",
        ":flexible_matrix_kernel",
        ":network_states",
        ":transition_system_traits",
        ":variable_store",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "feed_forward_network_kernel_test",
    size = "small",
    srcs = ["feed_forward_network_kernel_test.cc"],
    deps = [
        ":activation_functions",
        ":feed_forward_network_kernel",
        ":flexible_matrix_kernel",
        ":network_states",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "feed_forward_network",
    srcs = ["feed_forward_network.cc"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":extensions",
        ":feed_forward_network_kernel",
        ":feed_forward_network_layer",
        ":network_states",
        ":network_unit",
        ":network_unit_base",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "feed_forward_network_test",
    size = "small",
    srcs = ["feed_forward_network_test.cc"],
    deps = [
        ":dynamic_component",
        ":feed_forward_network",
        ":flexible_matrix_kernel",
        ":network_states",
        ":network_unit",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "bulk_feed_forward_network",
    srcs = ["bulk_feed_forward_network.cc"],
    copts = FAST_MATH_COPTS,
    opts_self = True,
    deps = [
        ":bulk_network_unit",
        ":extensions",
        ":feed_forward_network_kernel",
        ":feed_forward_network_layer",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "bulk_feed_forward_network_test",
    size = "small",
    srcs = ["bulk_feed_forward_network_test.cc"],
    deps = [
        ":bulk_feed_forward_network",
        ":bulk_network_unit",
        ":dynamic_component",
        ":flexible_matrix_kernel",
        ":network_states",
        ":variable_store",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        ":array_variable_store_builder",
        ":master",
        ":trained_model_variable_store",
        ":variable_store",
        ":variable_store_wrappers",
        "//dragnn/protos:runtime_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "conversion_test",
    size = "small",
    timeout = "moderate",
    srcs = ["conversion_test.cc"],
    data = [
        "testdata/conversion_output_variables_data",
        "testdata/conversion_output_variables_spec",
        ":test_rnn_tagger",
    ],
    shard_count = 6,
    deps = [
        ":conversion",
        ":dynamic_component",
        ":feed_forward_network",
        ":lstm_network",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core/test:generic",
        "//dragnn/protos:runtime_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "component_transformation",
    srcs = ["component_transformation.cc"],
    hdrs = ["component_transformation.h"],
    deps = [
        ":component",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "component_transformation_test",
    size = "small",
    srcs = ["component_transformation_test.cc"],
    deps = [
        ":component_transformation",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "fml_parsing",
    srcs = ["fml_parsing.cc"],
    hdrs = ["fml_parsing.h"],
    deps = [
        ":attributes",
        "//syntaxnet:base",
        "//syntaxnet:feature_extractor_proto_cc",
        "//syntaxnet:fml_parser",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "fml_parsing_test",
    size = "small",
    srcs = ["fml_parsing_test.cc"],
    deps = [
        ":fml_parsing",
        "//dragnn/core/test:generic",
        "//syntaxnet:base",
        "//syntaxnet:feature_extractor_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "term_map_utils",
    srcs = ["term_map_utils.cc"],
    hdrs = ["term_map_utils.h"],
    deps = [
        ":fml_parsing",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:feature_extractor_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "term_map_utils_test",
    size = "small",
    srcs = ["term_map_utils_test.cc"],
    deps = [
        ":term_map_utils",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "transition_system_traits",
    srcs = ["transition_system_traits.cc"],
    hdrs = ["transition_system_traits.h"],
    deps = [
        ":attributes",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "transition_system_traits_test",
    size = "small",
    srcs = ["transition_system_traits_test.cc"],
    deps = [
        ":transition_system_traits",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "unicode_dictionary",
    srcs = ["unicode_dictionary.cc"],
    hdrs = ["unicode_dictionary.h"],
    deps = [
        "//syntaxnet:base",
        "//syntaxnet:term_frequency_map",
        "//util/utf8:unicodetext",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "unicode_dictionary_test",
    size = "small",
    timeout = "moderate",
    srcs = ["unicode_dictionary_test.cc"],
    deps = [
        ":unicode_dictionary",
        "//dragnn/core/test:generic",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "//syntaxnet:term_frequency_map",
        "//third_party/utf",
        "//util/utf8:unicodetext",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sequence_extractor",
    srcs = ["sequence_extractor.cc"],
    hdrs = ["sequence_extractor.h"],
    deps = [
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "sequence_extractor_test",
    size = "small",
    srcs = ["sequence_extractor_test.cc"],
    deps = [
        ":sequence_extractor",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "term_map_sequence_extractor",
    hdrs = ["term_map_sequence_extractor.h"],
    deps = [
        ":sequence_extractor",
        ":term_map_utils",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:shared_store",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "term_map_sequence_extractor_test",
    size = "small",
    srcs = ["term_map_sequence_extractor_test.cc"],
    deps = [
        ":term_map_sequence_extractor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "//syntaxnet:term_frequency_map",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "syntaxnet_character_sequence_extractor",
    srcs = ["syntaxnet_character_sequence_extractor.cc"],
    deps = [
        ":sequence_extractor",
        ":term_map_sequence_extractor",
        ":term_map_utils",
        ":transition_system_traits",
        ":unicode_dictionary",
        "//dragnn/core:input_batch_cache",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:segmenter_utils",
        "//util/utf8:unicodetext",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "syntaxnet_character_sequence_extractor_test",
    size = "small",
    srcs = ["syntaxnet_character_sequence_extractor_test.cc"],
    deps = [
        ":sequence_extractor",
        ":syntaxnet_character_sequence_extractor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "syntaxnet_word_sequence_extractor",
    srcs = ["syntaxnet_word_sequence_extractor.cc"],
    deps = [
        ":sequence_extractor",
        ":term_map_sequence_extractor",
        ":term_map_utils",
        ":transition_system_traits",
        "//dragnn/core:input_batch_cache",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:term_frequency_map",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "syntaxnet_word_sequence_extractor_test",
    size = "small",
    srcs = ["syntaxnet_word_sequence_extractor_test.cc"],
    deps = [
        ":sequence_extractor",
        ":syntaxnet_word_sequence_extractor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "sequence_features",
    srcs = ["sequence_features.cc"],
    hdrs = ["sequence_features.h"],
    deps = [
        ":alignment",
        ":fixed_embeddings",
        ":sequence_extractor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "sequence_features_test",
    size = "small",
    srcs = ["sequence_features_test.cc"],
    deps = [
        ":fixed_embeddings",
        ":sequence_extractor",
        ":sequence_features",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sequence_linker",
    srcs = ["sequence_linker.cc"],
    hdrs = ["sequence_linker.h"],
    deps = [
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "sequence_linker_test",
    size = "small",
    srcs = ["sequence_linker_test.cc"],
    deps = [
        ":sequence_linker",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "identity_sequence_linker",
    srcs = ["identity_sequence_linker.cc"],
    deps = [
        ":sequence_linker",
        ":transition_system_traits",
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "identity_sequence_linker_test",
    size = "small",
    srcs = ["identity_sequence_linker_test.cc"],
    deps = [
        ":identity_sequence_linker",
        ":sequence_linker",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "reversed_sequence_linker",
    srcs = ["reversed_sequence_linker.cc"],
    deps = [
        ":sequence_linker",
        ":transition_system_traits",
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "reversed_sequence_linker_test",
    size = "small",
    srcs = ["reversed_sequence_linker_test.cc"],
    deps = [
        ":reversed_sequence_linker",
        ":sequence_linker",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "recurrent_sequence_linkers",
    srcs = ["recurrent_sequence_linkers.cc"],
    deps = [
        ":sequence_linker",
        ":transition_system_traits",
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "recurrent_sequence_linkers_test",
    size = "small",
    srcs = ["recurrent_sequence_linkers_test.cc"],
    deps = [
        ":recurrent_sequence_linkers",
        ":sequence_linker",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "syntaxnet_character_sequence_linkers",
    srcs = ["syntaxnet_character_sequence_linkers.cc"],
    deps = [
        ":sequence_linker",
        ":transition_system_traits",
        "//dragnn/core:input_batch_cache",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//util/utf8:unicodetext",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "syntaxnet_character_sequence_linkers_test",
    size = "small",
    srcs = ["syntaxnet_character_sequence_linkers_test.cc"],
    deps = [
        ":sequence_linker",
        ":syntaxnet_character_sequence_linkers",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "sequence_links",
    srcs = ["sequence_links.cc"],
    hdrs = ["sequence_links.h"],
    deps = [
        ":alignment",
        ":linked_embeddings",
        ":network_states",
        ":sequence_linker",
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "sequence_links_test",
    size = "small",
    srcs = ["sequence_links_test.cc"],
    deps = [
        ":linked_embeddings",
        ":network_states",
        ":sequence_linker",
        ":sequence_links",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sequence_predictor",
    srcs = ["sequence_predictor.cc"],
    hdrs = ["sequence_predictor.h"],
    deps = [
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "//syntaxnet:registry",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "sequence_predictor_test",
    size = "small",
    srcs = ["sequence_predictor_test.cc"],
    deps = [
        ":sequence_predictor",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "term_map_sequence_predictor",
    srcs = ["term_map_sequence_predictor.cc"],
    hdrs = ["term_map_sequence_predictor.h"],
    deps = [
        ":sequence_predictor",
        ":term_map_utils",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:shared_store",
        "//syntaxnet:term_frequency_map",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "term_map_sequence_predictor_test",
    size = "small",
    srcs = ["term_map_sequence_predictor_test.cc"],
    deps = [
        ":term_map_sequence_predictor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "syntaxnet_tag_sequence_predictor",
    srcs = ["syntaxnet_tag_sequence_predictor.cc"],
    deps = [
        ":sequence_predictor",
        ":term_map_sequence_predictor",
        ":transition_system_traits",
        "//dragnn/core:input_batch_cache",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "//syntaxnet:term_frequency_map",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "syntaxnet_tag_sequence_predictor_test",
    size = "small",
    srcs = ["syntaxnet_tag_sequence_predictor_test.cc"],
    deps = [
        ":alignment",
        ":sequence_predictor",
        ":syntaxnet_tag_sequence_predictor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/test:helpers",
        "//dragnn/runtime/test:term_map_helpers",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sequence_backend",
    srcs = ["sequence_backend.cc"],
    hdrs = ["sequence_backend.h"],
    deps = [
        "//dragnn/core:component_registry",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/interfaces:component",
        "//dragnn/core/interfaces:transition_state",
        "//dragnn/core/util:label",
        "//dragnn/protos:data_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sequence_backend_test",
    size = "small",
    srcs = ["sequence_backend_test.cc"],
    deps = [
        ":sequence_backend",
        "//dragnn/components/util:bulk_feature_extractor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/interfaces:transition_state",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "select_best_component_transformer",
    srcs = ["select_best_component_transformer.cc"],
    deps = [
        ":component",
        ":component_transformation",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "select_best_component_transformer_test",
    size = "small",
    srcs = ["select_best_component_transformer_test.cc"],
    deps = [
        ":component",
        ":component_transformation",
        ":extensions",
        ":select_best_component_transformer",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "sequence_component_transformer",
    srcs = ["sequence_component_transformer.cc"],
    deps = [
        ":component_transformation",
        ":sequence_extractor",
        ":sequence_linker",
        ":sequence_predictor",
        ":transition_system_traits",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sequence_component_transformer_test",
    size = "small",
    srcs = ["sequence_component_transformer_test.cc"],
    deps = [
        ":component_transformation",
        ":sequence_component_transformer",
        ":sequence_extractor",
        ":sequence_linker",
        ":sequence_predictor",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "stateless_component_transformer",
    srcs = ["stateless_component_transformer.cc"],
    deps = [
        ":component_transformation",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "stateless_component_transformer_test",
    size = "small",
    srcs = ["stateless_component_transformer_test.cc"],
    deps = [
        ":component_transformation",
        ":stateless_component_transformer",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "clear_dropout_component_transformer",
    srcs = ["clear_dropout_component_transformer.cc"],
    deps = [
        ":component_transformation",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:feature_extractor_proto_cc",
        "//syntaxnet:fml_parser",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "clear_dropout_component_transformer_test",
    size = "small",
    srcs = ["clear_dropout_component_transformer_test.cc"],
    deps = [
        ":clear_dropout_component_transformer",
        ":component_transformation",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "sequence_model",
    srcs = ["sequence_model.cc"],
    hdrs = ["sequence_model.h"],
    deps = [
        ":attributes",
        ":fixed_embeddings",
        ":linked_embeddings",
        ":network_states",
        ":sequence_backend",
        ":sequence_features",
        ":sequence_links",
        ":sequence_predictor",
        ":session_state",
        ":transition_system_traits",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

dragnn_cc_multiarch_test(
    name = "sequence_model_test",
    size = "small",
    srcs = ["sequence_model_test.cc"],
    deps = [
        ":fixed_embeddings",
        ":linked_embeddings",
        ":network_states",
        ":sequence_backend",
        ":sequence_extractor",
        ":sequence_linker",
        ":sequence_model",
        ":sequence_predictor",
        ":session_state",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

dragnn_cc_multiarch_library(
    name = "biaffine_digraph_component",
    srcs = ["biaffine_digraph_component.cc"],
    copts = FAST_MATH_COPTS,
    deps = [
        ":component",
        ":extensions",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime/math:eigen",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

dragnn_cc_multiarch_test(
    name = "biaffine_digraph_component_test",
    size = "small",
    srcs = ["biaffine_digraph_component_test.cc"],
    deps = [
        ":biaffine_digraph_component",
        ":component",
        ":extensions",
        ":network_states",
        ":session_state",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "head_selection_component_base",
    srcs = ["head_selection_component_base.cc"],
    hdrs = ["head_selection_component_base.h"],
    deps = [
        ":alignment",
        ":component",
        ":extensions",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "head_selection_component_base_test",
    size = "small",
    srcs = ["head_selection_component_base_test.cc"],
    deps = [
        ":head_selection_component_base",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "syntaxnet_head_selection_component",
    srcs = ["syntaxnet_head_selection_component.cc"],
    deps = [
        ":head_selection_component_base",
        ":session_state",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "syntaxnet_head_selection_component_test",
    size = "small",
    srcs = ["syntaxnet_head_selection_component_test.cc"],
    deps = [
        ":component",
        ":syntaxnet_head_selection_component",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "mst_solver_component_base",
    srcs = ["mst_solver_component_base.cc"],
    hdrs = ["mst_solver_component_base.h"],
    deps = [
        ":attributes",
        ":component",
        ":extensions",
        ":network_states",
        ":network_unit",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/mst:mst_solver",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "mst_solver_component_base_test",
    size = "small",
    srcs = ["mst_solver_component_base_test.cc"],
    deps = [
        ":mst_solver_component_base",
        ":network_states",
        ":session_state",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "syntaxnet_mst_solver_component",
    srcs = ["syntaxnet_mst_solver_component.cc"],
    deps = [
        ":mst_solver_component_base",
        ":session_state",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "syntaxnet_mst_solver_component_test",
    size = "small",
    srcs = ["syntaxnet_mst_solver_component_test.cc"],
    deps = [
        ":component",
        ":syntaxnet_mst_solver_component",
        ":variable_store",
        "//dragnn/core:compute_session",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/runtime/math:types",
        "//dragnn/runtime/test:network_test_base",
        "//syntaxnet:sentence_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "converter_main",
    srcs = ["converter.cc"],
    deps = [
        ":component_transformation",
        ":conversion",
        "//dragnn/runtime/myelin:myelination",
        "//dragnn/runtime/xla:xla_compilation",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
        "@sling//sling/base",
    ],
)

dragnn_cc_multiarch_binary(
    name = "converter",
    target_arch = "generic",
    deps = [
        ":biaffine_digraph_component",
        ":bulk_dynamic_component",
        ":bulk_feed_forward_network",
        ":bulk_lstm_network",
        ":clear_dropout_component_transformer",
        ":converter_main",
        ":dynamic_component",
        ":feed_forward_network",
        ":identity_sequence_linker",
        ":lstm_network",
        ":recurrent_sequence_linkers",
        ":reversed_sequence_linker",
        ":select_best_component_transformer",
        ":sequence_backend",
        ":sequence_bulk_dynamic_component",
        ":sequence_component_transformer",
        ":stateless_component_transformer",
        ":syntaxnet_character_sequence_extractor",
        ":syntaxnet_character_sequence_linkers",
        ":syntaxnet_head_selection_component",
        ":syntaxnet_mst_solver_component",
        ":syntaxnet_tag_sequence_predictor",
        ":syntaxnet_word_sequence_extractor",
        "//dragnn/components/stateless:stateless_component",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/mst:mst_ops_cc",
        "//dragnn/runtime/myelin:myelin_dynamic_component",
        "//dragnn/runtime/myelin:sequence_myelin_dynamic_component",
        "//dragnn/runtime/xla:xla_dynamic_component",
        "//syntaxnet:parser_transitions",
    ],
)

sh_test(
    name = "converter_test",
    size = "medium",
    srcs = ["converter_test.sh"],
    data = [":converter"] + glob([
        "testdata/converter_output/**",
        "testdata/rnn_tagger/**",
    ]),
)
