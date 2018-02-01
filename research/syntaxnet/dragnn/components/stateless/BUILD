package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

cc_library(
    name = "stateless_component",
    srcs = ["stateless_component.cc"],
    deps = [
        "//dragnn/core:component_registry",
        "//dragnn/core/interfaces:component",
        "//dragnn/core/interfaces:transition_state",
        "//dragnn/protos:data_proto",
        "//syntaxnet:base",
    ],
    alwayslink = 1,
)

cc_test(
    name = "stateless_component_test",
    srcs = ["stateless_component_test.cc"],
    deps = [
        ":stateless_component",
        "//dragnn/core:component_registry",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/test:generic",
        "//dragnn/core/test:mock_transition_state",
        "//dragnn/io:sentence_input_batch",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto",
        "//syntaxnet:test_main",
    ],
)
