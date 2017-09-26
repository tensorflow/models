local_repository(
  name = "org_tensorflow",
  path = "tensorflow",
)

# We need to pull in @io_bazel_rules_closure for TensorFlow. Bazel design
# documentation states that this verbosity is intentional, to prevent
# TensorFlow/SyntaxNet from depending on different versions of
# @io_bazel_rules_closure.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "25f5399f18d8bf9ce435f85c6bbf671ec4820bc4396b3022cc5dc4bc66303609",
    strip_prefix = "rules_closure-0.4.2",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/bazelbuild/rules_closure/archive/0.4.2.tar.gz",  # 2017-08-30
        "https://github.com/bazelbuild/rules_closure/archive/0.4.2.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(path_prefix="", tf_repo_name="org_tensorflow")

# Test that Bazel is up-to-date.
load("@org_tensorflow//tensorflow:workspace.bzl", "check_version")
check_version("0.4.2")

bind(
    name = "protobuf",
    actual = "@protobuf_archive//:protobuf",
)
