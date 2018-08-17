package(default_visibility = ["//visibility:public"])

cc_library(
    name = "disjoint_set_forest",
    hdrs = ["disjoint_set_forest.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "disjoint_set_forest_test",
    size = "small",
    srcs = ["disjoint_set_forest_test.cc"],
    deps = [
        ":disjoint_set_forest",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "spanning_tree_iterator",
    testonly = 1,
    srcs = ["spanning_tree_iterator.cc"],
    hdrs = ["spanning_tree_iterator.h"],
    deps = [
        "//syntaxnet:base",
    ],
)

cc_test(
    name = "spanning_tree_iterator_test",
    size = "small",
    srcs = ["spanning_tree_iterator_test.cc"],
    deps = [
        ":spanning_tree_iterator",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "mst_solver",
    hdrs = ["mst_solver.h"],
    deps = [
        ":disjoint_set_forest",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "mst_solver_test",
    size = "small",
    srcs = ["mst_solver_test.cc"],
    deps = [
        ":mst_solver",
        "//dragnn/core/test:generic",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "mst_solver_random_comparison_test",
    size = "small",
    timeout = "long",
    srcs = ["mst_solver_random_comparison_test.cc"],
    tags = [
        "manual",  # exclude from :all, since this is expensive
    ],
    deps = [
        ":mst_solver",
        ":spanning_tree_iterator",
        "//syntaxnet:base",
        "//syntaxnet:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)

tf_gen_op_libs(
    op_lib_names = ["mst_ops"],
)

# Don't use this library directly; instead use "dragnn/python:mst_ops".
tf_gen_op_wrapper_py(
    name = "mst_ops",
    visibility = ["//dragnn/python:__pkg__"],
    deps = [":mst_ops_op_lib"],
)

cc_library(
    name = "mst_ops_cc",
    srcs = [
        "ops/mst_op_kernels.cc",
        "ops/mst_ops.cc",
    ],
    deps = [
        ":mst_solver",
        "//syntaxnet:base",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)
