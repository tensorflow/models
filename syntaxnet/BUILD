# Description:
# A syntactic parser and part-of-speech tagger in TensorFlow.

package(
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//third_party/tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//third_party/tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

config_setting(
    name = "use_google_runtime",
    values = {"define": "tf_runtime=google"},
)

# proto libraries

proto_library(
    name = "feature_extractor_proto",
    srcs = ["feature_extractor.proto"],
    cc_api_version = 2,
    py_api_version = 2,
)

proto_library(
    name = "sentence_proto",
    srcs = ["sentence.proto"],
    cc_api_version = 2,
    py_api_version = 2,
)

py_proto_library(
    name = "sentence_proto_py_pb2",
    api_version = 2,
    deps = [":sentence_proto"],
)

proto_library(
    name = "dictionary_proto",
    srcs = ["dictionary.proto"],
    cc_api_version = 2,
    py_api_version = 2,
)

py_proto_library(
    name = "dictionary_proto_py_pb2",
    api_version = 2,
    deps = [":dictionary_proto"],
)

proto_library(
    name = "kbest_syntax_proto",
    srcs = ["kbest_syntax.proto"],
    cc_api_version = 2,
    py_api_version = 2,
    deps = [":sentence_proto"],
)

proto_library(
    name = "task_spec_proto",
    srcs = ["task_spec.proto"],
    cc_api_version = 2,
    py_api_version = 2,
)

py_proto_library(
    name = "task_spec_py_pb2",
    api_version = 2,
    deps = [":task_spec_proto"],
)

proto_library(
    name = "sparse_proto",
    srcs = ["sparse.proto"],
    cc_api_version = 2,
    py_api_version = 2,
)

py_proto_library(
    name = "sparse_py_pb2",
    api_version = 2,
    deps = [":sparse_proto"],
)

# cc libraries for feature extraction and parsing

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "document_format",
    srcs = ["document_format.cc"],
    hdrs = ["document_format.h"],
    deps = [
        ":registry",
        ":sentence_proto",
        ":task_context",
    ],
)

cc_library(
    name = "conll_syntax_format",
    srcs = ["conll_syntax_format.cc"],
    deps = [
        ":document_format",
    ],
    alwayslink = 1,
)

cc_library(
    name = "fml_parser",
    srcs = ["fml_parser.cc"],
    hdrs = ["fml_parser.h"],
    deps = [
        ":feature_extractor_proto",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "proto_io",
    hdrs = ["proto_io.h"],
    deps = [
        ":feature_extractor_proto",
        ":fml_parser",
        ":kbest_syntax_proto",
        ":sentence_proto",
        ":task_context",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "feature_extractor",
    srcs = ["feature_extractor.cc"],
    hdrs = ["feature_extractor.h"],
    deps = [
        ":feature_extractor_proto",
        ":fml_parser",
        ":kbest_syntax_proto",
        ":proto_io",
        ":sentence_proto",
        ":task_context",
        ":utils",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "feature_types",
    hdrs = ["feature_types.h"],
    deps = [
        ":feature_extractor",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "affix",
    srcs = ["affix.cc"],
    hdrs = ["affix.h"],
    deps = [
        ":dictionary_proto",
        ":feature_extractor",
        ":shared_store",
        ":term_frequency_map",
        ":utils",
        ":workspace",
    ],
)

cc_library(
    name = "sentence_features",
    srcs = ["sentence_features.cc"],
    hdrs = ["sentence_features.h"],
    deps = [
        ":affix",
        ":feature_extractor",
        ":feature_types",
        ":registry",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "shared_store",
    srcs = ["shared_store.cc"],
    hdrs = ["shared_store.h"],
    deps = [
        ":utils",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "workspace",
    srcs = ["workspace.cc"],
    hdrs = ["workspace.h"],
    deps = [
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "task_context",
    srcs = ["task_context.cc"],
    hdrs = ["task_context.h"],
    deps = [
        ":task_spec_proto",
        ":utils",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "term_frequency_map",
    srcs = ["term_frequency_map.cc"],
    hdrs = ["term_frequency_map.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "parser_transitions",
    srcs = [
        "arc_standard_transitions.cc",
        "parser_state.cc",
        "parser_transitions.cc",
        "tagger_transitions.cc",
    ],
    hdrs = [
        "parser_state.h",
        "parser_transitions.h",
    ],
    deps = [
        ":kbest_syntax_proto",
        ":registry",
        ":shared_store",
        ":task_context",
        ":term_frequency_map",
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "populate_test_inputs",
    testonly = 1,
    srcs = ["populate_test_inputs.cc"],
    hdrs = ["populate_test_inputs.h"],
    deps = [
        ":dictionary_proto",
        ":sentence_proto",
        ":task_context",
        ":term_frequency_map",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "parser_features",
    srcs = ["parser_features.cc"],
    hdrs = ["parser_features.h"],
    deps = [
        ":affix",
        ":feature_extractor",
        ":parser_transitions",
        ":registry",
        ":sentence_features",
        ":sentence_proto",
        ":task_context",
        ":term_frequency_map",
        ":workspace",
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "embedding_feature_extractor",
    srcs = ["embedding_feature_extractor.cc"],
    hdrs = ["embedding_feature_extractor.h"],
    deps = [
        ":feature_extractor",
        ":parser_features",
        ":parser_transitions",
        ":sparse_proto",
        ":task_context",
        ":workspace",
    ],
)

cc_library(
    name = "parser_state_context",
    srcs = ["parser_state_context.cc"],
    hdrs = ["parser_state_context.h"],
    deps = [
        ":embedding_feature_extractor",
        ":feature_extractor",
        ":parser_features",
        ":parser_transitions",
        ":sparse_proto",
        ":task_context",
        ":task_spec_proto",
        ":term_frequency_map",
        ":workspace",
    ],
)

cc_library(
    name = "reader_ops",
    srcs = [
        "beam_reader_ops.cc",
        "reader_ops.cc",
    ],
    deps = [
        ":parser_features",
        ":parser_state_context",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":task_spec_proto",
        "//third_party/eigen3",
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "document_filters",
    srcs = ["document_filters.cc"],
    deps = [
        ":conll_syntax_format",
        ":document_format",
        ":parser_features",
        ":parser_state_context",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":task_spec_proto",
        "//third_party/eigen3",
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lexicon_builder",
    srcs = ["lexicon_builder.cc"],
    deps = [
        ":conll_syntax_format",
        ":document_format",
        ":parser_features",
        ":parser_state_context",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":task_spec_proto",
        "//third_party/eigen3",
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unpack_sparse_features",
    srcs = ["unpack_sparse_features.cc"],
    deps = [
        ":sparse_proto",
        "//third_party/eigen3",
        "//third_party/tensorflow/core",
    ],
)

cc_library(
    name = "parser_ops_cc",
    srcs = ["ops/parser_ops.cc"],
    deps = [
        "//third_party/tensorflow/core",
    ],
    alwayslink = 1,
)

# cc tests

filegroup(
    name = "testdata",
    srcs = [
        "testdata/context.pbtxt",
        "testdata/document",
        "testdata/document2",
        "testdata/wsj-train-100",
    ],
)

cc_test(
    name = "shared_store_test",
    size = "small",
    srcs = ["shared_store_test.cc"],
    deps = [
        ":shared_store",
        "//testing/base/public:gunit_main",
    ],
)

cc_test(
    name = "sentence_features_test",
    size = "medium",
    srcs = ["sentence_features_test.cc"],
    deps = [
        ":feature_extractor",
        ":populate_test_inputs",
        ":sentence_features",
        ":sentence_proto",
        ":task_context",
        ":task_spec_proto",
        ":term_frequency_map",
        ":workspace",
        "//testing/base/public:gunit_main",
    ],
)

cc_test(
    name = "arc_standard_transitions_test",
    size = "small",
    srcs = ["arc_standard_transitions_test.cc"],
    data = [":testdata"],
    deps = [
        ":parser_transitions",
        ":populate_test_inputs",
        "//testing/base/public:gunit_main",
    ],
)

cc_test(
    name = "tagger_transitions_test",
    size = "small",
    srcs = ["tagger_transitions_test.cc"],
    data = [":testdata"],
    deps = [
        ":parser_transitions",
        ":populate_test_inputs",
        "//testing/base/public:gunit_main",
    ],
)

cc_test(
    name = "parser_features_test",
    size = "small",
    srcs = ["parser_features_test.cc"],
    deps = [
        ":feature_extractor",
        ":parser_features",
        ":parser_transitions",
        ":populate_test_inputs",
        ":sentence_proto",
        ":task_context",
        ":task_spec_proto",
        ":term_frequency_map",
        ":workspace",
        "//testing/base/public:gunit_main",
    ],
)

# py graph builder and trainer

tf_gen_op_libs(
    op_lib_names = ["parser_ops"],
)

tf_gen_op_wrapper_py(
    name = "parser_ops",
    deps = [":parser_ops_op_lib"],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    deps = [
        ":parser_ops",
    ] + select({
        ":use_google_runtime": ["//third_party/py/tensorflow:tensorflow_google"],
        "//conditions:default": ["//third_party/py/tensorflow"],
    }),
)

py_library(
    name = "structured_graph_builder",
    srcs = ["structured_graph_builder.py"],
    deps = [
        ":graph_builder",
        ":parser_ops",
    ] + select({
        ":use_google_runtime": ["//third_party/py/tensorflow:tensorflow_google"],
        "//conditions:default": ["//third_party/py/tensorflow"],
    }),
)

py_binary(
    name = "parser_trainer",
    srcs = ["parser_trainer.py"],
    deps = [
        ":document_filters",
        ":graph_builder",
        ":lexicon_builder",
        ":parser_ops",
        ":parser_ops_cc",
        ":reader_ops",
        ":structured_graph_builder",
        ":task_spec_py_pb2",
        ":unpack_sparse_features",
        "//file/colossus/cns",
    ] + select({
        "//third_party/gpus/cuda:cuda_crosstool_condition": [
            "//learning/brain/public:tensorflow_gpu_deps",
        ],
        "//conditions:default": [],
    }),
)

py_binary(
    name = "parser_eval",
    srcs = ["parser_eval.py"],
    deps = [
        ":document_filters",
        ":graph_builder",
        ":lexicon_builder",
        ":parser_ops",
        ":parser_ops_cc",
        ":reader_ops",
        ":sentence_proto_py_pb2",
        ":structured_graph_builder",
        ":unpack_sparse_features",
        "//file/colossus/cns",
    ],
)

# py tests

py_test(
    name = "reader_ops_test",
    size = "medium",
    srcs = ["reader_ops_test.py"],
    data = [":testdata"],
    tags = ["notsan"],
    deps = [
        ":dictionary_proto_py_pb2",
        ":graph_builder",
        ":lexicon_builder",
        ":parser_ops",
        ":parser_ops_cc",
        ":reader_ops",
        ":sparse_py_pb2",
        ":unpack_sparse_features",
    ],
)

py_test(
    name = "beam_reader_ops_test",
    size = "medium",
    srcs = ["beam_reader_ops_test.py"],
    data = [":testdata"],
    tags = ["notsan"],
    deps = [
        ":lexicon_builder",
        ":parser_ops",
        ":parser_ops_cc",
        ":reader_ops",
        ":structured_graph_builder",
        ":unpack_sparse_features",
    ],
)

py_test(
    name = "graph_builder_test",
    size = "medium",
    srcs = ["graph_builder_test.py"],
    data = [":testdata"],
    tags = ["notsan"],
    deps = [
        ":graph_builder",
        ":lexicon_builder",
        ":parser_ops",
        ":parser_ops_cc",
        ":reader_ops",
        ":sparse_py_pb2",
        ":unpack_sparse_features",
    ],
)
