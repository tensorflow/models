package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "synthetic_transit_maker",
    srcs = [
        "synthetic_transit_maker.py",
    ],
)

py_test(
    name = "synthetic_transit_maker_test",
    srcs = ["synthetic_transit_maker_test.py"],
    srcs_version = "PY2AND3",
    deps = [":synthetic_transit_maker"],
)
