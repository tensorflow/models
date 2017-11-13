package(default_visibility = ["//visibility:public"])

load(
    "//syntaxnet:syntaxnet.bzl",
    "tf_proto_library",
    "tf_proto_library_py",
)

# Protos.

tf_proto_library(
    name = "data_proto",
    srcs = ["data.proto"],
)

tf_proto_library(
    name = "trace_proto",
    srcs = ["trace.proto"],
    deps = [
        ":data_proto",
    ],
)

tf_proto_library(
    name = "spec_proto",
    srcs = ["spec.proto"],
)

tf_proto_library(
    name = "runtime_proto",
    srcs = ["runtime.proto"],
    deps = [":spec_proto"],
)

tf_proto_library_py(
    name = "data_py_pb2",
    srcs = ["data.proto"],
)

tf_proto_library_py(
    name = "trace_py_pb2",
    srcs = ["trace.proto"],
    deps = [":data_py_pb2"],
)

tf_proto_library_py(
    name = "spec_py_pb2",
    srcs = ["spec.proto"],
)
