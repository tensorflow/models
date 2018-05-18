package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

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
        "//dragnn/core/util:label",
        "//dragnn/io:sentence_input_batch",
        "//dragnn/io:syntaxnet_sentence",
        "//dragnn/protos:data_proto_cc",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:parser_transitions",
        "//syntaxnet:registry",
        "//syntaxnet:sparse_proto_cc",
        "//syntaxnet:task_context",
        "//syntaxnet:task_spec_proto_cc",
        "//syntaxnet:utils",
        "//util/utf8:unicodetext",
    ],
    alwayslink = 1,
)

cc_library(
    name = "syntaxnet_link_feature_extractor",
    srcs = ["syntaxnet_link_feature_extractor.cc"],
    hdrs = ["syntaxnet_link_feature_extractor.h"],
    deps = [
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:embedding_feature_extractor",
        "//syntaxnet:parser_transitions",
        "//syntaxnet:task_context",
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
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:parser_transitions",
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
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "//syntaxnet:test_main",
    ],
)

cc_test(
    name = "syntaxnet_link_feature_extractor_test",
    srcs = ["syntaxnet_link_feature_extractor_test.cc"],
    deps = [
        ":syntaxnet_link_feature_extractor",
        "//dragnn/core/test:generic",
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:task_context",
        "//syntaxnet:test_main",
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
        "//dragnn/protos:spec_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
        "//syntaxnet:test_main",
    ],
)
