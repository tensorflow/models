package(default_visibility = ["//visibility:public"])

cc_library(
    name = "syntaxnet_component",
    srcs = ["syntaxnet_component.cc"],
    hdrs = ["syntaxnet_component.h"],
    deps = [
        ":syntaxnet_link_feature_extractor",
        ":syntaxnet_transition_state",
        "//dragnn/components/util:bulk_feature_extractor",
        "//dragnn/core:beam",
        "//dragnn/core:component_registry",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/interfaces:component",
        "//dragnn/core/interfaces:transition_state",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:data_proto",
        "//dragnn/protos:spec_proto",
        "//dragnn/protos:trace_proto",
        "//syntaxnet:base",
        "//syntaxnet:parser_transitions",
        "//syntaxnet:registry",
        "//syntaxnet:sparse_proto",
        "//syntaxnet:task_context",
        "//syntaxnet:task_spec_proto",
        "//syntaxnet:utils",
        "@org_tensorflow//tensorflow/core:lib",  # For tf/core/platform/logging.h
    ],
    alwayslink = 1,
)

cc_library(
    name = "syntaxnet_link_feature_extractor",
    srcs = ["syntaxnet_link_feature_extractor.cc"],
    hdrs = ["syntaxnet_link_feature_extractor.h"],
    deps = [
        "//dragnn/protos:spec_proto",
        "//syntaxnet:embedding_feature_extractor",
        "//syntaxnet:parser_transitions",
        "//syntaxnet:task_context",
        "@org_tensorflow//tensorflow/core:lib",  # For tf/core/platform/logging.h
    ],
)

cc_library(
    name = "syntaxnet_transition_state",
    srcs = ["syntaxnet_transition_state.cc"],
    hdrs = ["syntaxnet_transition_state.h"],
    deps = [
        "//dragnn/core/interfaces:cloneable_transition_state",
        "//dragnn/core/interfaces:transition_state",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:trace_proto",
        "//syntaxnet:base",
        "//syntaxnet:parser_transitions",
        "@org_tensorflow//tensorflow/core:lib",  # For tf/core/platform/logging.h
    ],
)

# Test data.
filegroup(
    name = "testdata",
    data = glob(["testdata/**"]),
)

# Tests.
cc_test(
    name = "syntaxnet_component_test",
    srcs = ["syntaxnet_component_test.cc"],
    data = [":testdata"],
    deps = [
        ":syntaxnet_component",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/core/test:mock_transition_state",
        "//dragnn/io:sentence_input_batch",
        "//syntaxnet:sentence_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "syntaxnet_link_feature_extractor_test",
    srcs = ["syntaxnet_link_feature_extractor_test.cc"],
    deps = [
        ":syntaxnet_link_feature_extractor",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto",
        "//syntaxnet:task_context",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_test(
    name = "syntaxnet_transition_state_test",
    srcs = ["syntaxnet_transition_state_test.cc"],
    data = [":testdata"],
    deps = [
        ":syntaxnet_component",
        ":syntaxnet_transition_state",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/core/test:mock_transition_state",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/protos:spec_proto",
        "//syntaxnet:sentence_proto",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)
