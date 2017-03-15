# Description:
# A syntactic parser and part-of-speech tagger in TensorFlow.

package(
    default_visibility = [
        "//visibility:public",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load(
    "syntaxnet",
    "tf_proto_library",
    "tf_proto_library_py",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)

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
    name = "sentence_py_pb2",
    srcs = ["sentence.proto"],
)

tf_proto_library(
    name = "dictionary_proto",
    srcs = ["dictionary.proto"],
)

tf_proto_library_py(
    name = "dictionary_py_pb2",
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
        "@com_googlesource_code_re2//:re2",
        "@protobuf//:protobuf",
        "@org_tensorflow//third_party/eigen3",
    ] + select({
        "//conditions:default": [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
        ],
        "@org_tensorflow//tensorflow:darwin": [
            "@org_tensorflow//tensorflow/core:framework_headers_lib",
        ],
    }),
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
        "//external:gtest",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
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
    name = "text_formats",
    srcs = ["text_formats.cc"],
    deps = [
        ":base",
        ":document_format",
        ":segmenter_utils",
        ":sentence_proto",
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
        ":sentence_proto",
        ":task_context",
    ],
)

cc_library(
    name = "char_properties",
    srcs = ["char_properties.cc"],
    hdrs = ["char_properties.h"],
    deps = [
        ":registry",
        ":utils",
        "//util/utf8:unicodetext",
    ],
    alwayslink = 1,
)

cc_library(
    name = "char_ngram_string_extractor",
    srcs = ["char_ngram_string_extractor.cc"],
    hdrs = ["char_ngram_string_extractor.h"],
    deps = [
        ":segmenter_utils",
        ":task_context",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "segmenter_utils",
    srcs = ["segmenter_utils.cc"],
    hdrs = ["segmenter_utils.h"],
    deps = [
        ":base",
        ":char_properties",
        ":sentence_proto",
        "//util/utf8:unicodetext",
    ],
    alwayslink = 1,
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
        ":sentence_proto",
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
        ":char_ngram_string_extractor",
        ":char_properties",
        ":feature_extractor",
        ":registry",
        ":segmenter_utils",
        ":shared_store",
        ":task_context",
        ":workspace",
        "//util/utf8:unicodetext",
    ],
    alwayslink = 1,
)

cc_library(
    name = "whole_sentence_features",
    srcs = ["whole_sentence_features.cc"],
    hdrs = ["whole_sentence_features.h"],
    deps = [
        ":base",
        ":feature_extractor",
        ":registry",
        ":task_context",
        ":workspace",
    ],
    alwayslink = 1,
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
    name = "morphology_label_set",
    srcs = ["morphology_label_set.cc"],
    hdrs = ["morphology_label_set.h"],
    deps = [
        ":document_format",
        ":feature_extractor",
        ":proto_io",
        ":registry",
        ":sentence_proto",
        ":utils",
    ],
)

cc_library(
    name = "parser_transitions",
    srcs = [
        "arc_standard_transitions.cc",
        "binary_segment_state.cc",
        "binary_segment_transitions.cc",
        "char_shift_transitions.cc",
        "head_transitions.cc",
        "head_transitions.h",
        "label_transitions.cc",
        "label_transitions.h",
        "morpher_transitions.cc",
        "once_transitions.cc",
        "parser_features.cc",
        "parser_state.cc",
        "parser_transitions.cc",
        "shift_transitions.cc",
        "tagger_transitions.cc",
    ],
    hdrs = [
        "binary_segment_state.h",
        "char_shift_transitions.h",
        "parser_features.h",
        "parser_state.h",
        "parser_transitions.h",
    ],
    deps = [
        ":base",
        ":feature_extractor",
        ":morphology_label_set",
        ":registry",
        ":segmenter_utils",
        ":sentence_features",
        ":sentence_proto",
        ":shared_store",
        ":task_context",
        ":term_frequency_map",
        ":utils",
        ":whole_sentence_features",
        ":workspace",
        "@org_tensorflow//tensorflow/core:lib",
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
        ":task_spec_proto",
        ":term_frequency_map",
        ":test_main",
    ],
)

cc_library(
    name = "embedding_feature_extractor",
    srcs = ["embedding_feature_extractor.cc"],
    hdrs = ["embedding_feature_extractor.h"],
    deps = [
        ":feature_extractor",
        ":parser_transitions",
        ":sentence_features",
        ":sparse_proto",
        ":task_context",
        ":utils",
        ":workspace",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "sentence_batch",
    srcs = ["sentence_batch.cc"],
    hdrs = ["sentence_batch.h"],
    deps = [
        ":embedding_feature_extractor",
        ":feature_extractor",
        ":parser_transitions",
        ":sentence_proto",
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
        ":parser_transitions",
        ":sentence_batch",
        ":sentence_proto",
        ":sparse_proto",
        ":task_context",
        ":task_spec_proto",
    ],
    alwayslink = 1,
)

cc_library(
    name = "document_filters",
    srcs = ["document_filters.cc"],
    deps = [
        ":document_format",
        ":parser_transitions",
        ":segmenter_utils",
        ":sentence_batch",
        ":sentence_proto",
        ":task_context",
        ":text_formats",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lexicon_builder",
    srcs = ["lexicon_builder.cc"],
    deps = [
        ":affix",
        ":char_ngram_string_extractor",
        ":feature_extractor",
        ":parser_transitions",
        ":segmenter_utils",
        ":sentence_batch",
        ":sentence_proto",
        ":term_frequency_map",
        ":text_formats",
        ":utils",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
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
    alwayslink = 1,
)

cc_library(
    name = "parser_ops_cc",
    srcs = ["ops/parser_ops.cc"],
    deps = [
        ":document_filters",
        ":lexicon_builder",
        ":reader_ops",
        ":unpack_sparse_features",
        "@org_tensorflow//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "parser_ops.so",
    linkopts = select({
        "//conditions:default": ["-lm"],
        "@org_tensorflow//tensorflow:darwin": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":parser_ops_cc",
    ],
)

# cc tests

filegroup(
    name = "testdata",
    srcs = [
        "testdata/context.pbtxt",
        "testdata/document",
        "testdata/mini-training-set",
    ],
)

filegroup(
    name = "parsey_data",
    srcs = glob(["models/parsey_mcparseface/*"]),
)

cc_test(
    name = "binary_segment_state_test",
    size = "small",
    srcs = ["binary_segment_state_test.cc"],
    deps = [
        ":base",
        ":parser_transitions",
        ":term_frequency_map",
        ":test_main",
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
    name = "char_properties_test",
    srcs = ["char_properties_test.cc"],
    deps = [
        ":char_properties",
        ":test_main",
    ],
)

cc_test(
    name = "char_ngram_string_extractor_test",
    srcs = ["char_ngram_string_extractor_test.cc"],
    deps = [
        ":char_ngram_string_extractor",
        ":task_context",
        ":test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "segmenter_utils_test",
    srcs = ["segmenter_utils_test.cc"],
    deps = [
        ":base",
        ":segmenter_utils",
        ":sentence_proto",
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
        ":utils",
        ":workspace",
    ],
)

cc_test(
    name = "whole_sentence_features_test",
    size = "medium",
    srcs = ["whole_sentence_features_test.cc"],
    deps = [
        ":feature_extractor",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":term_frequency_map",
        ":test_main",
        ":whole_sentence_features",
        ":workspace",
    ],
)

cc_test(
    name = "morphology_label_set_test",
    srcs = ["morphology_label_set_test.cc"],
    deps = [
        ":morphology_label_set",
        ":test_main",
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
        ":sentence_proto",
        ":task_spec_proto",
        ":test_main",
    ],
)

cc_test(
    name = "char_shift_transitions_test",
    size = "small",
    srcs = ["char_shift_transitions_test.cc"],
    data = [":testdata"],
    deps = [
        ":parser_transitions",
        ":populate_test_inputs",
        ":sentence_proto",
        ":task_spec_proto",
        ":test_main",
    ],
)

cc_test(
    name = "binary_segment_transitions_test",
    size = "small",
    srcs = ["binary_segment_transitions_test.cc"],
    deps = [
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":test_main",
        ":workspace",
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
        ":sentence_proto",
        ":task_spec_proto",
        ":test_main",
    ],
)

cc_test(
    name = "once_transitions_test",
    size = "small",
    srcs = ["once_transitions_test.cc"],
    deps = [
        ":base",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":term_frequency_map",
        ":test_main",
    ],
)

cc_test(
    name = "head_transitions_test",
    size = "small",
    srcs = ["head_transitions_test.cc"],
    deps = [
        ":base",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":term_frequency_map",
        ":test_main",
    ],
)

cc_test(
    name = "label_transitions_test",
    size = "small",
    srcs = ["label_transitions_test.cc"],
    deps = [
        ":base",
        ":parser_transitions",
        ":sentence_proto",
        ":task_context",
        ":term_frequency_map",
        ":test_main",
    ],
)

cc_test(
    name = "parser_features_test",
    size = "small",
    srcs = ["parser_features_test.cc"],
    deps = [
        ":feature_extractor",
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
    deps = [":parser_ops_op_lib"],
)

py_library(
    name = "load_parser_ops_py",
    srcs = ["load_parser_ops.py"],
    data = [":parser_ops.so"],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    deps = [
        ":load_parser_ops_py",
        ":parser_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "structured_graph_builder",
    srcs = ["structured_graph_builder.py"],
    deps = [
        ":graph_builder",
    ],
)

py_binary(
    name = "parser_trainer",
    srcs = ["parser_trainer.py"],
    deps = [
        ":graph_builder",
        ":structured_graph_builder",
        ":task_spec_py_pb2",
    ],
)

py_binary(
    name = "parser_eval",
    srcs = ["parser_eval.py"],
    deps = [
        ":graph_builder",
        ":sentence_py_pb2",
        ":structured_graph_builder",
        ":task_spec_py_pb2",
    ],
)

py_binary(
    name = "conll2tree",
    srcs = ["conll2tree.py"],
    deps = [
        ":graph_builder",
        ":sentence_py_pb2",
    ],
)

# py tests

py_test(
    name = "lexicon_builder_test",
    size = "small",
    srcs = ["lexicon_builder_test.py"],
    deps = [
        ":graph_builder",
        ":sentence_py_pb2",
        ":task_spec_py_pb2",
    ],
)

py_test(
    name = "text_formats_test",
    size = "small",
    srcs = ["text_formats_test.py"],
    deps = [
        ":graph_builder",
        ":sentence_py_pb2",
        ":task_spec_py_pb2",
    ],
)

py_test(
    name = "reader_ops_test",
    size = "medium",
    srcs = ["reader_ops_test.py"],
    data = [":testdata"],
    tags = ["notsan"],
    deps = [
        ":dictionary_py_pb2",
        ":graph_builder",
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
        ":structured_graph_builder",
    ],
)

py_test(
    name = "graph_builder_test",
    size = "medium",
    srcs = ["graph_builder_test.py"],
    data = [
        ":testdata",
    ],
    tags = ["notsan"],
    deps = [
        ":graph_builder",
        ":sparse_py_pb2",
    ],
)

sh_test(
    name = "parser_trainer_test",
    size = "large",
    srcs = ["parser_trainer_test.sh"],
    data = [
        ":parser_eval",
        ":parser_trainer",
        ":testdata",
    ],
    tags = ["slow"],
)
