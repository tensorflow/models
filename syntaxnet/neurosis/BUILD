# Description:
# A syntactic parser and part-of-speech tagger in TensorFlow.
#
# To build:
#   git clone --recursive rpc://team/saft/neurosis
#   cd neurosis
#   tensorflow/configure
#   bazel test neurosis/...
#
# Requires bazel and swig to be installed.

package(
    default_visibility = ["//visibility:private"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load(
    "neurosis",
    "tf_proto_library",
    "tf_proto_library_py",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)

# config_setting(
#     name = "use_google_runtime",
#     values = {"define": "tf_runtime=google"},
# )

# proto libraries

tf_proto_library(
    name = "feature_extractor_proto",
    srcs = ["feature_extractor.proto"],
)

tf_proto_library(
    name = "sentence_proto",
    srcs = ["sentence.proto"],
)

tf_proto_library_py(
    name = "sentence_proto_py_pb2",
    srcs = ["sentence.proto"],
)

tf_proto_library(
    name = "dictionary_proto",
    srcs = ["dictionary.proto"],
)

tf_proto_library_py(
    name = "dictionary_proto_py_pb2",
    srcs = ["dictionary.proto"],
)

tf_proto_library(
    name = "kbest_syntax_proto",
    srcs = ["kbest_syntax.proto"],
    deps = [":sentence_proto"],
)

tf_proto_library(
    name = "task_spec_proto",
    srcs = ["task_spec.proto"],
)

tf_proto_library_py(
    name = "task_spec_py_pb2",
    srcs = ["task_spec.proto"],
)

tf_proto_library(
    name = "sparse_proto",
    srcs = ["sparse.proto"],
)

tf_proto_library_py(
    name = "sparse_py_pb2",
    srcs = ["sparse.proto"],
)

# cc libraries for feature extraction and parsing

cc_library(
    name = "base",
    hdrs = ["base.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        "@tf//third_party/eigen3",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = [
        "utils.h",
    ],
    deps = [
        ":base",
        "//util/utf8:unicodetext",
    ],
)

cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    linkopts = ["-lm"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:testlib",
        "//external:gtest",
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
        ":utils",
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
    ],
)

cc_library(
    name = "feature_extractor",
    srcs = ["feature_extractor.cc"],
    hdrs = [
        "feature_extractor.h",
        "feature_types.h",
    ],
    deps = [
        ":document_format",
        ":feature_extractor_proto",
        ":kbest_syntax_proto",
        ":proto_io",
        ":sentence_proto",
        ":task_context",
        ":utils",
        ":workspace",
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
        ":registry",
    ],
)

cc_library(
    name = "shared_store",
    srcs = ["shared_store.cc"],
    hdrs = ["shared_store.h"],
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "workspace",
    srcs = ["workspace.cc"],
    hdrs = ["workspace.h"],
    deps = [
        ":utils",
    ],
)

cc_library(
    name = "task_context",
    srcs = ["task_context.cc"],
    hdrs = ["task_context.h"],
    deps = [
        ":task_spec_proto",
        ":utils",
    ],
)

cc_library(
    name = "term_frequency_map",
    srcs = ["term_frequency_map.cc"],
    hdrs = ["term_frequency_map.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
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
        ":test_main",
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
    ],
    alwayslink = 1,
)

cc_library(
    name = "unpack_sparse_features",
    srcs = ["unpack_sparse_features.cc"],
    deps = [
        ":sparse_proto",
        ":utils",
    ],
)

cc_library(
    name = "parser_ops_cc",
    srcs = ["ops/parser_ops.cc"],
    deps = [
        ":base",
        ":document_filters",
        ":lexicon_builder",
        ":parser_ops_op_lib",
        ":reader_ops",
        ":unpack_sparse_features",
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
        ":test_main",
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
        ":test_main",
        ":workspace",
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
        ":test_main",
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
        ":test_main",
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
        ":test_main",
        ":workspace",
    ],
)

# py graph builder and trainer

tf_gen_op_libs(
    op_lib_names = ["parser_ops"],
)

tf_gen_op_wrapper_py(
    name = "parser_ops",
    deps = [
        ":document_filters",
        ":lexicon_builder",
        ":parser_ops_op_lib",
        ":reader_ops",
        ":unpack_sparse_features",
    ],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    deps = [
        ":parser_ops",
    ],
    # + select({
    #     ":use_google_runtime": ["//third_party/py/tensorflow:tensorflow_google"],
    #     "//conditions:default": ["//third_party/py/tensorflow"],
    # }),
)

py_library(
    name = "structured_graph_builder",
    srcs = ["structured_graph_builder.py"],
    deps = [
        ":graph_builder",
        ":parser_ops",
    ],
    # + select({
    #     ":use_google_runtime": ["//third_party/py/tensorflow:tensorflow_google"],
    #     "//conditions:default": ["//third_party/py/tensorflow"],
    # }),
)

py_binary(
    name = "parser_trainer",
    srcs = ["parser_trainer.py"],
    deps = [
        ":graph_builder",
        ":parser_ops",
        ":structured_graph_builder",
        ":task_spec_py_pb2",
        #"//file/colossus/cns",
    ],
    # + select({
    #     "//third_party/gpus/cuda:cuda_crosstool_condition": [
    #         "//learning/brain/public:tensorflow_gpu_deps",
    #     ],
    #     "//conditions:default": [],
    # }),
)

py_binary(
    name = "parser_eval",
    srcs = ["parser_eval.py"],
    deps = [
        ":graph_builder",
        ":parser_ops",
        ":sentence_proto_py_pb2",
        ":structured_graph_builder",
        #"//file/colossus/cns",
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
        "@tf//tensorflow:tensorflow_py",
        ":dictionary_proto_py_pb2",
        ":graph_builder",
        ":parser_ops",
        ":sparse_py_pb2",
    ],
)

py_test(
    name = "beam_reader_ops_test",
    size = "medium",
    srcs = ["beam_reader_ops_test.py"],
    data = [":testdata"],
    tags = ["notsan"],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":parser_ops",
        ":structured_graph_builder",
    ],
)

py_test(
    name = "graph_builder_test",
    size = "medium",
    srcs = ["graph_builder_test.py"],
    data = [":testdata"],
    tags = ["notsan"],
    deps = [
        "@tf//tensorflow:tensorflow_py",
        ":graph_builder",
        ":parser_ops",
        ":sparse_py_pb2",
    ],
)
