# TensorFlow Serving external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

def tf_serving_workspace():
  '''All TensorFlow Serving external dependencies.'''
  # The inception model's BUILD file is written as if it is the root BUILD
  # file. We use strip_prefix to make the inception model directory the root.
  native.http_archive(
      name = "inception_model",
      urls = [
          "https://mirror.bazel.build/github.com/tensorflow/models/archive/6fc65ee60ac39be0445e5a311b40dc7ccce214d0.tar.gz",
          "https://github.com/tensorflow/models/archive/6fc65ee60ac39be0445e5a311b40dc7ccce214d0.tar.gz",
      ],
      sha256 = "7a908017d60fca54c80405527576f08dbf8d130efe6a53791639ff3b26afffbc",
      strip_prefix = "models-6fc65ee60ac39be0445e5a311b40dc7ccce214d0/research/inception",
  )

  tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

  # ===== gRPC dependencies =====
  native.bind(
      name = "libssl",
      actual = "@boringssl//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )

  # gRPC wants the existence of a cares dependence but its contents are not
  # actually important since we have set GRPC_ARES=0 in tools/bazel.rc
  native.bind(
      name = "cares",
      actual = "@grpc//third_party/nanopb:nanopb",
  )
