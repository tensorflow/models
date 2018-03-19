package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "generate_input_records",
    srcs = ["generate_input_records.py"],
    deps = [":preprocess"],
)

py_binary(
    name = "preprocess",
    srcs = ["preprocess.py"],
    deps = [
        "//light_curve_util:kepler_io",
        "//light_curve_util:median_filter",
        "//light_curve_util:util",
        "//third_party/kepler_spline",
    ],
)
