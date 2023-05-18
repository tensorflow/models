licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "sgnn_projection",
    srcs = ["sgnn_projection.cc"],
    hdrs = ["sgnn_projection.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:context",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@farmhash_archive//:farmhash",
        "@flatbuffers",
    ],
)

cc_library(
    name = "sgnn_projection_op_resolver",
    srcs = ["sgnn_projection_op_resolver.cc"],
    hdrs = ["sgnn_projection_op_resolver.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sgnn_projection",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sgnn_projection_test",
    srcs = ["sgnn_projection_test.cc"],
    deps = [
        ":sgnn_projection",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

py_library(
    name = "sgnn",
    srcs = [
        "sgnn.py",
    ],
    srcs_version = "PY3",
    deps = [
        # package tensorflow
        "@org_tflite_support//tensorflow_lite_support/custom_ops/python:tflite_text_api",
        # Expect tensorflow text installed
    ],
)

py_test(
    name = "sgnn_test",
    srcs = [
        "sgnn_test.py",
    ],
    deps = [
        ":sgnn",
        # package tensorflow
        # Expect tensorflow text installed
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    main = "train.py",
    python_version = "PY3",
    deps = [
        ":sgnn",
        # package tensorflow
        # package tensorflow_datasets
    ],
)

py_binary(
    name = "run_tflite",
    srcs = ["run_tflite.py"],
    main = "run_tflite.py",
    python_version = "PY3",
    deps = [
        ":sgnn_projection_op_resolver",
        # Expect numpy installed
        # package TFLite flex delegate
        # package TFLite interpreter
        "@org_tflite_support//tensorflow_lite_support/custom_ops/kernel:_pywrap_ngrams_op_resolver",
        "@org_tflite_support//tensorflow_lite_support/custom_ops/kernel:_pywrap_whitespace_tokenizer_op_resolver",
        # Expect tensorflow text installed
    ],
)

# pip install numpy
py_library(
    name = "expect_numpy_installed",
)

# pip install tensroflow_text
py_library(
    name = "expect_tensorflow_text_installed",
)
