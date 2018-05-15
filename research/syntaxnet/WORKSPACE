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
    sha256 = "6691c58a2cd30a86776dd9bb34898b041e37136f2dc7e24cadaeaf599c95c657",
    strip_prefix = "rules_closure-08039ba8ca59f64248bb3b6ae016460fe9c9914f",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

http_archive(
    name = "sling",
    sha256 = "f1ce597476cb024808ca0a371a01db9dda4e0c58fb34a4f9c4ea91796f437b10",
    strip_prefix = "sling-e3ae9d94eb1d9ee037a851070d54ed2eefaa928a",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/google/sling/archive/e3ae9d94eb1d9ee037a851070d54ed2eefaa928a.tar.gz",
        "https://github.com/google/sling/archive/e3ae9d94eb1d9ee037a851070d54ed2eefaa928a.tar.gz",
    ],
)

# Used by SLING.
bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)
