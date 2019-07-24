package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "mobile_ssd_client",
    srcs = ["mobile_ssd_client.cc"],
    hdrs = ["mobile_ssd_client.h"],
    deps = [
        "//protos:box_encodings_cc_proto",
        "//protos:detections_cc_proto",
        "//protos:labelmap_cc_proto",
        "//protos:mobile_ssd_client_options_cc_proto",
        "//utils:conversion_utils",
        "//utils:ssd_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:span",
        "@com_google_glog//:glog",
        "@gemmlowp",
    ],
)

cc_library(
    name = "mobile_ssd_tflite_client",
    srcs = ["mobile_ssd_tflite_client.cc"],
    hdrs = ["mobile_ssd_tflite_client.h"],
    deps = [
        ":mobile_ssd_client",
        "//protos:anchor_generation_options_cc_proto",
        "//utils:file_utils",
        "//utils:ssd_utils",
        "@com_google_absl//absl/memory",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/lite:arena_planner",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mobile_lstd_tflite_client",
    srcs = ["mobile_lstd_tflite_client.cc"],
    hdrs = ["mobile_lstd_tflite_client.h"],
    deps = [
        ":mobile_ssd_client",
        ":mobile_ssd_tflite_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_glog//:glog",
    ],
    alwayslink = 1,
)
