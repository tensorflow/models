# Tensorflow Object Detection API: Matcher implementations.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0
py_library(
    name = "argmax_matcher",
    srcs = [
        "argmax_matcher.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:matcher",
        "//tensorflow/models/research/object_detection/utils:shape_utils",
    ],
)

py_test(
    name = "argmax_matcher_test",
    srcs = ["argmax_matcher_test.py"],
    deps = [
        ":argmax_matcher",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:test_case",
    ],
)

py_library(
    name = "bipartite_matcher",
    srcs = [
        "bipartite_matcher.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/contrib/image:image_py",
        "//tensorflow/models/research/object_detection/core:matcher",
    ],
)

py_test(
    name = "bipartite_matcher_test",
    srcs = [
        "bipartite_matcher_test.py",
    ],
    deps = [
        ":bipartite_matcher",
        "//tensorflow",
    ],
)
