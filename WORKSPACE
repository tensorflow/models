local_repository(
  name = "tf",
  path = __workspace_dir__ + "/tensorflow",
)

load('//tensorflow/tensorflow:workspace.bzl', 'tf_workspace')
tf_workspace("tensorflow/")
# grpc expects //external:protobuf_clib and //external:protobuf_compiler
# to point to the protobuf's compiler library.
bind(
    name = "protobuf_clib",
    actual = "@tf//google/protobuf:protoc_lib",
)

bind(
    name = "protobuf_compiler",
    actual = "@tf//google/protobuf:protoc_lib",
)

git_repository(
    name = "grpc",
    commit = "73979f4",
    init_submodules = True,
    remote = "https://github.com/grpc/grpc.git",
)

# protobuf expects //external:grpc_cpp_plugin to point to grpc's
# C++ plugin code generator.
bind(
    name = "grpc_cpp_plugin",
    actual = "@grpc//:grpc_cpp_plugin",
)

bind(
    name = "grpc_lib",
    actual = "@grpc//:grpc++_unsecure",
)
