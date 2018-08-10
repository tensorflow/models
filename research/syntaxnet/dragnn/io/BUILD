package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sentence_input_batch",
    srcs = ["sentence_input_batch.cc"],
    hdrs = ["sentence_input_batch.h"],
    deps = [
        ":syntaxnet_sentence",
        "//dragnn/core/interfaces:input_batch",
        "//syntaxnet:base",
        "//syntaxnet:sentence_proto_cc",
    ],
)

cc_library(
    name = "syntaxnet_sentence",
    hdrs = ["syntaxnet_sentence.h"],
    deps = [
        "//syntaxnet:sentence_proto_cc",
        "//syntaxnet:workspace",
    ],
)

cc_test(
    name = "sentence_input_batch_test",
    srcs = ["sentence_input_batch_test.cc"],
    deps = [
        ":sentence_input_batch",
        "//dragnn/core/test:generic",
        "//syntaxnet:sentence_proto_cc",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
