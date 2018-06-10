# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
load("//build_rules:repo.bzl", "tensorflow_http_archive")

tensorflow_http_archive(
  name = "org_tensorflow",
  sha256 = "cbd96914936ce3aacc39e02c2efb711f937f8ebcda888c349eab075185d7c914",
  git_commit = "d8fac4cb80eb0c42d2550bcb720a80d29fc5f22d",
  #sha256 = "783341cb4190db39166cd3ffb3b1fc590f93b7c5f95539819776f52cfebcb1ff",
  #git_commit = "d752244fbaad5e4268243355046d30990f59418f",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
  name = "io_bazel_rules_closure",
  sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
  strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
  urls = [
    "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
    "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
  ],
)

# Abseil
#http_archive(
#  name = "com_google_absl",
#  urls = ["https://github.com/abseil/abseil-cpp/archive/master.zip"],
#  strip_prefix = "abseil-cpp-master",
#)

# Please add all new TensorFlow Serving dependencies in workspace.bzl.
load("//build_rules:workspace.bzl", "tf_serving_workspace")

tf_serving_workspace()
