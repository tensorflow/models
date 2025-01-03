package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

proto_library(
    name = "box_encodings_proto",
    srcs = ["box_encodings.proto"],
)

cc_proto_library(
    name = "box_encodings_cc_proto",
    deps = [":box_encodings_proto"],
)

proto_library(
    name = "detections_proto",
    srcs = ["detections.proto"],
    deps = [":box_encodings_proto"],
)

cc_proto_library(
    name = "detections_cc_proto",
    deps = [":detections_proto"],
)

proto_library(
    name = "labelmap_proto",
    srcs = ["labelmap.proto"],
)

cc_proto_library(
    name = "labelmap_cc_proto",
    deps = [":labelmap_proto"],
)

proto_library(
    name = "mobile_ssd_client_options_proto",
    srcs = ["mobile_ssd_client_options.proto"],
    deps = [
        ":anchor_generation_options_proto",
        ":box_encodings_proto",
        ":labelmap_proto",
    ],
)

cc_proto_library(
    name = "mobile_ssd_client_options_cc_proto",
    deps = [":mobile_ssd_client_options_proto"],
)

proto_library(
    name = "anchor_generation_options_proto",
    srcs = ["anchor_generation_options.proto"],
)

cc_proto_library(
    name = "anchor_generation_options_cc_proto",
    deps = [":anchor_generation_options_proto"],
)
