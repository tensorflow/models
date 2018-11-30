"""A TensorFlow model for generative modeling of light curves."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":astrowavenet_model",
        ":configurations",
        "//astrowavenet/data:kepler_light_curves",
        "//astrowavenet/data:synthetic_transits",
        "//astrowavenet/util:estimator_util",
        "//tf_util:config_util",
        "//tf_util:configdict",
        "//tf_util:estimator_runner",
    ],
)

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "astrowavenet_model",
    srcs = [
        "astrowavenet_model.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "astrowavenet_model_test",
    size = "small",
    srcs = [
        "astrowavenet_model_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":astrowavenet_model",
        ":configurations",
        "//tf_util:configdict",
    ],
)
