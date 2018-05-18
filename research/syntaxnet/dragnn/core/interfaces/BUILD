package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

cc_library(
    name = "cloneable_transition_state",
    hdrs = ["cloneable_transition_state.h"],
    deps = [":transition_state"],
)

cc_library(
    name = "component",
    hdrs = ["component.h"],
    deps = [
        ":transition_state",
        "//dragnn/components/util:bulk_feature_extractor",
        "//dragnn/core:input_batch_cache",
        "//dragnn/core/util:label",
        "//dragnn/protos:spec_proto_cc",
        "//dragnn/protos:trace_proto_cc",
        "//syntaxnet:base",
        "//syntaxnet:registry",
    ],
)

cc_library(
    name = "input_batch",
    hdrs = ["input_batch.h"],
    deps = [
        "//syntaxnet:base",
    ],
)

cc_library(
    name = "transition_state",
    hdrs = ["transition_state.h"],
    deps = [
        "//syntaxnet:base",
    ],
)
