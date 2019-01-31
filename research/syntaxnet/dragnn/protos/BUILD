package(default_visibility = ["//visibility:public"])

load(
    "//syntaxnet:syntaxnet.bzl",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)

# Protos.

tf_proto_library_cc(
    name = "data_proto",
    srcs = ["data.proto"],
)

tf_proto_library_cc(
    name = "trace_proto",
    srcs = ["trace.proto"],
    protodeps = [":data_proto"],
)

tf_proto_library_cc(
    name = "cell_trace_proto",
    srcs = ["cell_trace.proto"],
    protodeps = [":trace_proto"],
)

tf_proto_library_cc(
    name = "spec_proto",
    srcs = ["spec.proto"],
)

tf_proto_library_cc(
    name = "runtime_proto",
    srcs = ["runtime.proto"],
    protodeps = [":spec_proto"],
)

tf_proto_library_cc(
    name = "export_proto",
    srcs = ["export.proto"],
    protodeps = [":spec_proto"],
)

tf_proto_library_py(
    name = "data_pb2",
    srcs = ["data.proto"],
)

tf_proto_library_py(
    name = "trace_pb2",
    srcs = ["trace.proto"],
    protodeps = [":data_pb2"],
)

tf_proto_library_py(
    name = "spec_pb2",
    srcs = ["spec.proto"],
)

tf_proto_library_py(
    name = "export_pb2",
    srcs = ["export.proto"],
)
