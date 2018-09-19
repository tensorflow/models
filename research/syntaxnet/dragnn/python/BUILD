package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "dragnn_cc_impl.so",
    linkopts = select({
        "//conditions:default": ["-lm"],
        "@org_tensorflow//tensorflow:darwin": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//dragnn/components/stateless:stateless_component",
        "//dragnn/components/syntaxnet:syntaxnet_component",
        "//dragnn/core:dragnn_bulk_ops_cc",
        "//dragnn/core:dragnn_ops_cc",
    ],
)

cc_binary(
    name = "mst_cc_impl.so",
    linkopts = select({
        "//conditions:default": ["-lm"],
        "@org_tensorflow//tensorflow:darwin": [],
    }),
    linkshared = 1,
    linkstatic = 1,
    deps = ["//dragnn/mst:mst_ops_cc"],
)

filegroup(
    name = "testdata",
    data = glob(["testdata/**"]),
)

py_library(
    name = "load_dragnn_cc_impl_py",
    srcs = ["load_dragnn_cc_impl.py"],
    data = [":dragnn_cc_impl.so"],
)

py_library(
    name = "load_mst_cc_impl_py",
    srcs = ["load_mst_cc_impl.py"],
    data = [":mst_cc_impl.so"],
)

py_library(
    name = "bulk_component",
    srcs = [
        "bulk_component.py",
    ],
    deps = [
        ":dragnn_ops",
        ":network_units",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "components",
    srcs = [
        "component.py",
    ],
    deps = [
        ":bulk_component",
        ":dragnn_ops",
        ":network_units",
        ":runtime_support",
        "//dragnn/protos:export_pb2_py",
        "//syntaxnet/util:check",
        "//syntaxnet/util:pyregistry",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "composite_optimizer",
    srcs = ["composite_optimizer.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "dragnn_ops",
    srcs = ["dragnn_ops.py"],
    deps = [
        ":load_dragnn_cc_impl_py",
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//syntaxnet:load_parser_ops_py",
    ],
)

py_library(
    name = "dragnn_model_saver_lib",
    srcs = ["dragnn_model_saver_lib.py"],
    deps = [
        ":dragnn_ops",
        ":graph_builder",
        ":load_dragnn_cc_impl_py",
        ":network_units",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:sentence_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "dragnn_model_saver_lib_test",
    srcs = ["dragnn_model_saver_lib_test.py"],
    data = [":testdata"],
    deps = [
        ":dragnn_model_saver_lib",
        "//dragnn/protos:export_pb2_py",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:test_flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "dragnn_model_saver",
    srcs = ["dragnn_model_saver.py"],
    deps = [
        ":dragnn_model_saver_lib",
        ":spec_builder",
        "//dragnn/protos:spec_pb2_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "graph_builder",
    srcs = ["graph_builder.py"],
    deps = [
        ":biaffine_units",
        ":components",
        ":composite_optimizer",
        ":dragnn_ops",
        ":network_units",
        ":transformer_units",
        ":wrapped_units",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "network_units",
    srcs = ["network_units.py"],
    deps = [
        ":dragnn_ops",
        "//syntaxnet:syntaxnet_ops",
        "//syntaxnet/util:check",
        "//syntaxnet/util:pyregistry",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "render_parse_tree_graphviz",
    srcs = ["render_parse_tree_graphviz.py"],
    deps = [
    ],
)

py_test(
    name = "render_parse_tree_graphviz_test",
    srcs = ["render_parse_tree_graphviz_test.py"],
    deps = [
        ":render_parse_tree_graphviz",
        "//syntaxnet:sentence_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "render_spec_with_graphviz",
    srcs = ["render_spec_with_graphviz.py"],
    deps = [
        "//dragnn/protos:spec_pb2_py",
    ],
)

py_test(
    name = "render_spec_with_graphviz_test",
    srcs = ["render_spec_with_graphviz_test.py"],
    deps = [
        ":render_spec_with_graphviz",
        ":spec_builder",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sentence_io",
    srcs = ["sentence_io.py"],
    deps = [
        "//syntaxnet:parser_ops",
    ],
)

py_binary(
    name = "visualization",
    srcs = ["visualization.py"],
    data = [
        "//dragnn/viz:viz-min-js-gz",
    ],
    deps = [
        "//dragnn/protos:trace_pb2_py",
    ],
)

py_test(
    name = "visualization_test",
    srcs = ["visualization_test.py"],
    deps = [
        ":visualization",
        "//dragnn/protos:spec_pb2_py",
        "//dragnn/protos:trace_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "wrapped_units",
    srcs = ["wrapped_units.py"],
    deps = [
        ":network_units",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

# Tests

py_test(
    name = "component_test",
    srcs = [
        "component_test.py",
    ],
    deps = [
        ":components",
        "//dragnn/protos:spec_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "bulk_component_test",
    srcs = [
        "bulk_component_test.py",
    ],
    deps = [
        ":bulk_component",
        ":components",
        ":dragnn_ops",
        ":network_units",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:sentence_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "composite_optimizer_test",
    srcs = ["composite_optimizer_test.py"],
    deps = [
        ":composite_optimizer",
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//syntaxnet:load_parser_ops_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "graph_builder_test",
    size = "large",
    srcs = ["graph_builder_test.py"],
    data = [
        "//dragnn/core:testdata",
    ],
    shard_count = 5,
    tags = [
        "notsan",
    ],
    deps = [
        ":dragnn_ops",
        ":graph_builder",
        "//dragnn/protos:spec_pb2_py",
        "//dragnn/protos:trace_pb2_py",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:sentence_pb2_py",
        "//syntaxnet:test_flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "network_units_test",
    size = "small",
    srcs = ["network_units_test.py"],
    deps = [
        ":network_units",
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:load_parser_ops_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "sentence_io_test",
    srcs = ["sentence_io_test.py"],
    data = ["//syntaxnet:testdata"],
    deps = [
        ":dragnn_ops",
        ":sentence_io",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:parser_ops",
        "//syntaxnet:sentence_pb2_py",
        "//syntaxnet:test_flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer_lib.py"],
    deps = [
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:parser_ops",
        "//syntaxnet:sentence_pb2_py",
        "//syntaxnet:task_spec_pb2_py",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "trainer_lib_test",
    srcs = ["trainer_lib_test.py"],
    deps = [
        ":trainer_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "lexicon",
    srcs = ["lexicon.py"],
    deps = [
        "//syntaxnet:parser_ops",
        "//syntaxnet:task_spec_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "lexicon_test",
    srcs = ["lexicon_test.py"],
    deps = [
        ":lexicon",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:parser_ops",
        "//syntaxnet:parser_trainer",
        "//syntaxnet:test_flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        "//syntaxnet:sentence_pb2_py",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    deps = [
        ":evaluation",
        "//syntaxnet:sentence_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "spec_builder",
    srcs = ["spec_builder.py"],
    deps = [
        ":lexicon",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:parser_ops",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "spec_builder_test",
    srcs = ["spec_builder_test.py"],
    deps = [
        ":spec_builder",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:load_parser_ops_py",
        "//syntaxnet:parser_ops",
        "//syntaxnet:parser_trainer",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "digraph_ops",
    srcs = ["digraph_ops.py"],
    deps = [
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "digraph_ops_test",
    srcs = ["digraph_ops_test.py"],
    deps = [
        ":digraph_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "biaffine_units",
    srcs = ["biaffine_units.py"],
    deps = [
        ":digraph_ops",
        ":network_units",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "biaffine_units_test",
    srcs = ["biaffine_units_test.py"],
    deps = [
        ":biaffine_units",
        ":network_units",
        "//dragnn/protos:spec_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "transformer_units",
    srcs = ["transformer_units.py"],
    deps = [
        ":network_units",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "transformer_units_test",
    size = "small",
    srcs = ["transformer_units_test.py"],
    deps = [
        ":network_units",
        ":transformer_units",
        "//dragnn/core:dragnn_bulk_ops",
        "//dragnn/core:dragnn_ops",
        "//dragnn/protos:spec_pb2_py",
        "//syntaxnet:load_parser_ops_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_library(
    name = "runtime_support",
    srcs = ["runtime_support.py"],
    deps = [
        ":network_units",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "runtime_support_test",
    srcs = ["runtime_support_test.py"],
    deps = [
        ":network_units",
        ":runtime_support",
        "//dragnn/protos:export_pb2_py",
        "//dragnn/protos:spec_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "file_diff_test",
    srcs = ["file_diff_test.py"],
    deps = [
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mst_ops",
    srcs = ["mst_ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":digraph_ops",
        ":load_mst_cc_impl_py",
        "//dragnn/mst:mst_ops",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "mst_ops_test",
    srcs = ["mst_ops_test.py"],
    deps = [
        ":mst_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mst_units",
    srcs = ["mst_units.py"],
    deps = [
        ":mst_ops",
        ":network_units",
        "//syntaxnet/util:check",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "mst_units_test",
    size = "small",
    srcs = ["mst_units_test.py"],
    deps = [
        ":mst_units",
        ":network_units",
        "//dragnn/protos:spec_pb2_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
