package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "conversion_utils",
    srcs = ["conversion_utils.cc"],
    hdrs = ["conversion_utils.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_glog//:glog",
    ],
)

cc_test(
    name = "conversion_utils_test",
    srcs = ["conversion_utils_test.cc"],
    deps = [
        ":conversion_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ssd_utils",
    srcs = ["ssd_utils.cc"],
    hdrs = ["ssd_utils.h"],
    deps = [
        "//protos:anchor_generation_options_cc_proto",
        "//protos:box_encodings_cc_proto",
        "//protos:detections_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "file_utils",
    srcs = ["file_utils.cc"],
    hdrs = ["file_utils.h"],
    deps = [
        "//protos:labelmap_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
    ],
)
