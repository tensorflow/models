package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "generate_input_records",
    srcs = ["generate_input_records.py"],
    deps = [":preprocess"],
)

py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        "//light_curve:kepler_io",
        "//light_curve:median_filter",
        "//light_curve:util",
        "//tf_util:example_util",
        "//third_party/kepler_spline",
    ],
)
