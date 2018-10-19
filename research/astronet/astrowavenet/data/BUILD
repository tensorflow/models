package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "base",
    srcs = [
        "base.py",
    ],
    deps = [
        "//astronet/ops:dataset_ops",
        "//astronet/util:configdict",
    ],
)

py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    data = ["test_data/test-dataset.tfrecord"],
    srcs_version = "PY2AND3",
    deps = [":base"],
)

py_library(
    name = "kepler_light_curves",
    srcs = [
        "kepler_light_curves.py",
    ],
    deps = [
        ":base",
        "//astronet/util:configdict",
    ],
)

py_library(
    name = "synthetic_transits",
    srcs = [
        "synthetic_transits.py",
    ],
    deps = [
        ":base",
        ":synthetic_transit_maker",
        "//astronet/util:configdict",
    ],
)

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
