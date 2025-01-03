# Tensorflow ops for sequence string projection.

load("//tf_ops:build_def.bzl", "gen_op_wrapper_py")

licenses(["notice"])

package(
    default_visibility = [
        "//:__subpackages__",
    ],
)

cc_library(
    name = "projection_normalizer_util",
    srcs = ["projection_normalizer_util.cc"],
    hdrs = ["projection_normalizer_util.h"],
    deps = [
        ":projection_util",
        "@icu4c",
    ],
)

cc_library(
    name = "projection_tokenizer_util",
    srcs = ["projection_tokenizer_util.cc"],
    hdrs = ["projection_tokenizer_util.h"],
    deps = [
        ":projection_util",
    ],
)

cc_library(
    name = "projection_util",
    srcs = ["projection_util.cc"],
    hdrs = ["projection_util.h"],
    deps = [
        "@icu4c",
    ],
)

cc_library(
    name = "skipgram_finder",
    srcs = ["skipgram_finder.cc"],
    hdrs = ["skipgram_finder.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@icu4c",
    ],
)

cc_test(
    name = "skipgram_finder_test",
    srcs = ["skipgram_finder_test.cc"],
    deps = [
        ":skipgram_finder",
        "@com_google_absl//absl/strings",
	"@com_google_googletest//:gtest_main",
        "@icu4c",
    ],
)

cc_library(
    name = "subsequence_finder",
    srcs = ["subsequence_finder.cc"],
    hdrs = ["subsequence_finder.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@icu4c",
    ],
)

cc_test(
    name = "subsequence_finder_test",
    srcs = ["subsequence_finder_test.cc"],
    deps = [
        ":subsequence_finder",
	"@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "text_distorter",
    srcs = ["text_distorter.cc"],
    hdrs = ["text_distorter.h"],
    deps = [
        "@icu4c",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
)

cc_library(
    name = "denylist_op",
    srcs = ["denylist_op.cc"],
    deps = [
        ":skipgram_finder",
        ":subsequence_finder",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

gen_op_wrapper_py(
    name = "denylist_op_py",
    out = "denylist_op.py",
    kernel_lib = ":denylist_op",
)

py_test(
    name = "denylist_op_py_test",
    srcs = ["denylist_op_test.py"],
    main = "denylist_op_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":denylist_op_py",
    ],
)

cc_library(
    name = "sequence_string_projection_op",
    srcs = [
        "sequence_string_projection.cc",
    ],
    deps = [
        ":projection_normalizer_util",
        ":projection_tokenizer_util",
        ":projection_util",
        ":text_distorter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sequence_string_projection_test",
    size = "small",
    srcs = ["sequence_string_projection_test.cc"],
    deps = [
        ":sequence_string_projection_op",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
)

gen_op_wrapper_py(
    name = "sequence_string_projection_op_py",
    out = "sequence_string_projection_op.py",
    kernel_lib = ":sequence_string_projection_op",
)

cc_library(
    name = "sequence_string_projection_op_v2",
    srcs = [
        "sequence_string_projection_op_v2.cc",
    ],
    deps = [
        ":projection_normalizer_util",
        ":projection_util",
        ":text_distorter",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sequence_string_projection_op_v2_test",
    size = "small",
    srcs = ["sequence_string_projection_op_v2_test.cc"],
    deps = [
        ":sequence_string_projection_op_v2",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
)

gen_op_wrapper_py(
    name = "sequence_string_projection_op_v2_py",
    out = "sequence_string_projection_op_v2.py",
    kernel_lib = ":sequence_string_projection_op_v2",
)

cc_library(
    name = "tf_custom_ops",
    srcs = ["tf_custom_ops.cc"],
    deps = [
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

gen_op_wrapper_py(
    name = "tf_custom_ops_py",
    out = "tf_custom_ops_py.py",
    kernel_lib = ":tf_custom_ops",
)
