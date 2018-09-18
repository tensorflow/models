package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

py_library(
    name = "kepler_spline",
    srcs = ["kepler_spline.py"],
    srcs_version = "PY2AND3",
    deps = ["//third_party/robust_mean"],
)

py_test(
    name = "kepler_spline_test",
    size = "small",
    srcs = ["kepler_spline_test.py"],
    srcs_version = "PY2AND3",
    deps = [":kepler_spline"],
)
