# Tensorflow ops for sequence string projection.

load("//tf_ops:build_def.bzl", "gen_op_wrapper_py")

licenses(["notice"])

package(
    default_visibility = [
        "//:__subpackages__",
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
        "@com_google_absl//absl/random",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "projection_util",
    srcs = ["projection_util.cc"],
    hdrs = ["projection_util.h"],
    deps = [
        "@utf_archive//:utf",
    ],
)

cc_library(
    name = "projection_tokenizer_util",
    srcs = ["projection_tokenizer_util.cc"],
    hdrs = ["projection_tokenizer_util.h"],
    deps = [
        ":projection_util",
        "@utf_archive//:utf",
    ],
)

cc_library(
    name = "projection_normalizer_util",
    srcs = ["projection_normalizer_util.cc"],
    hdrs = ["projection_normalizer_util.h"],
    deps = [
        ":projection_util",
        "@utf_archive//:utf",
    ],
)

cc_library(
    name = "text_distorter",
    srcs = ["text_distorter.cc"],
    hdrs = ["text_distorter.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@icu4c",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
        "@utf_archive//:utf",
    ],
)

cc_test(
    name = "sequence_string_projection_test",
    size = "small",
    srcs = ["sequence_string_projection_test.cc"],
    deps = [
        ":sequence_string_projection_op",
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
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
        "@tensorflow_includes//:includes",
        "@tensorflow_solib//:framework_lib",
    ],
)

gen_op_wrapper_py(
    name = "sequence_string_projection_op_v2_py",
    out = "sequence_string_projection_op_v2.py",
    kernel_lib = ":sequence_string_projection_op_v2",
)

gen_op_wrapper_py(
    name = "sequence_string_projection_op_py",
    out = "sequence_string_projection_op.py",
    kernel_lib = ":sequence_string_projection_op",
)
