licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "sgnn_projection",
    srcs = ["sgnn_projection.cc"],
    hdrs = ["sgnn_projection.h"],
    deps = [
        "@com_github_google_flatbuffers//:flatbuffers",
        # package farmhash:farmhash_lib
        "//third_party/tensorflow/lite:context",
        "//third_party/tensorflow/lite:string_util",
        "//third_party/tensorflow/lite/kernels:kernel_util",
        "//third_party/tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_library(
    name = "sgnn_projection_op_resolver",
    srcs = ["sgnn_projection_op_resolver.cc"],
    hdrs = ["sgnn_projection_op_resolver.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sgnn_projection",
        "//third_party/tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "sgnn_projection_test",
    srcs = ["sgnn_projection_test.cc"],
    deps = [
        ":sgnn_projection",
        "//testing/base/public:gunit_main",
        "@com_github_google_flatbuffers//:flatbuffers",
        "//third_party/tensorflow/lite:string_util",
        "//third_party/tensorflow/lite/kernels:test_util",
        "//third_party/tensorflow/lite/schema:schema_fbs",
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
        # Expect tensorflow text installed",
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
        # Expect tensorflow text installed",
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
        # Expect numpy installed",
        "//third_party/tensorflow/lite/delegates/flex:delegate",
        "//third_party/tensorflow/lite/python:interpreter",
        "@org_tflite_support//tensorflow_lite_support/custom_ops/kernel:ngrams_op_resolver",
        "@org_tflite_support//tensorflow_lite_support/custom_ops/kernel:whitespace_tokenizer_op_resolver",
        # Expect tensorflow text installed",
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
