package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "median",
    hdrs = ["median.h"],
)

cc_test(
    name = "median_test",
    size = "small",
    srcs = [
        "median_test.cc",
    ],
    deps = [
        ":median",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "median_filter",
    srcs = ["median_filter.cc"],
    hdrs = ["median_filter.h"],
    deps = [
        ":median",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "median_filter_test",
    size = "small",
    srcs = [
        "median_filter_test.cc",
    ],
    deps = [
        ":median_filter",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "phase_fold",
    srcs = ["phase_fold.cc"],
    hdrs = ["phase_fold.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_test(
    name = "phase_fold_test",
    size = "small",
    srcs = [
        "phase_fold_test.cc",
    ],
    deps = [
        ":phase_fold",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "normalize",
    srcs = ["normalize.cc"],
    hdrs = ["normalize.h"],
    deps = [
        ":median",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "normalize_test",
    size = "small",
    srcs = [
        "normalize_test.cc",
    ],
    deps = [
        ":normalize",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "view_generator",
    srcs = ["view_generator.cc"],
    hdrs = ["view_generator.h"],
    deps = [
        ":median_filter",
        ":normalize",
        ":phase_fold",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "view_generator_test",
    size = "small",
    srcs = [
        "view_generator_test.cc",
    ],
    deps = [
        ":test_util",
        ":view_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    hdrs = ["test_util.h"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)
