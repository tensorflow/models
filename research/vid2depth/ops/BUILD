load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_data",
    srcs = glob(["testdata/**"]),
)

cc_library(
    name = "icp_op_kernel",
    srcs = ["icp_op_kernel.cc"],
    copts = [
        "-fexceptions",
        "-Wno-sign-compare",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    deps = [
        "@com_github_pointcloudlibrary_pcl//:common",
        "@com_github_pointcloudlibrary_pcl//:registration",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_custom_op_library(
    name = "icp_op.so",
    linkopts = ["-llz4"],
    deps = [
        ":icp_op_kernel",
    ],
)

py_library(
    name = "icp_op",
    srcs = ["icp_op.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_library(
    name = "icp_util",
    srcs = ["icp_util.py"],
    data = [":test_data"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "icp_grad",
    srcs = ["icp_grad.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
    ],
)

cc_binary(
    name = "pcl_demo",
    srcs = ["pcl_demo.cc"],
    deps = [
        "@com_github_pointcloudlibrary_pcl//:common",
        "@com_github_pointcloudlibrary_pcl//:registration",
    ],
)

py_binary(
    name = "icp_train_demo",
    srcs = ["icp_train_demo.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
        ":icp_grad",
        ":icp_util",
    ],
)

py_test(
    name = "icp_test",
    size = "small",
    srcs = ["icp_test.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
        ":icp_util",
    ],
)

py_test(
    name = "icp_grad_test",
    size = "small",
    srcs = ["icp_grad_test.py"],
    data = [
        ":icp_op.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        ":icp_op",
        ":icp_grad",
        ":icp_test",
    ],
)
