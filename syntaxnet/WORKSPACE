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
    sha256 = "60fc6977908f999b23ca65698c2bb70213403824a84f7904310b6000d78be9ce",
    strip_prefix = "rules_closure-5ca1dab6df9ad02050f7ba4e816407f88690cf7d",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",  # 2017-02-03
        "https://github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(path_prefix="", tf_repo_name="org_tensorflow")

# Test that Bazel is up-to-date.
load("@org_tensorflow//tensorflow:workspace.bzl", "check_version")
check_version("0.4.2")
