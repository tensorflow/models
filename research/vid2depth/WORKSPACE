workspace(name = "vid2depth")

# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
load(":repo.bzl", "tensorflow_http_archive")

tensorflow_http_archive(
    name = "org_tensorflow",
    git_commit = "bc69c4ceed6544c109be5693eb40ddcf3a4eb95d",
    sha256 = "21d6ac553adcfc9d089925f6d6793fee6a67264a0ce717bc998636662df4ca7e",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "dbe0da2cca88194d13dc5a7125a25dd7b80e1daec7839f33223de654d7a1bcc8",
    strip_prefix = "rules_closure-ba3e07cb88be04a2d4af7009caa0ff3671a79d06",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/ba3e07cb88be04a2d4af7009caa0ff3671a79d06.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/ba3e07cb88be04a2d4af7009caa0ff3671a79d06.tar.gz",  # 2017-10-31
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)

bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
)

bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)

# gRPC wants the existence of a cares dependence but its contents are not
# actually important since we have set GRPC_ARES=0 in tools/bazel.rc
bind(
    name = "cares",
    actual = "@grpc//third_party/nanopb:nanopb",
)

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:workspace.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("0.5.4")

# TODO(rodrigoq): rename to com_github_antonovvk_bazel_rules to match cartographer.
http_archive(
    name = "bazel_rules",
    sha256 = "b6e1b6cfc17f676c70045deb6d46bb330490693e65c8d541aae265ea34a48c8c",
    strip_prefix = "bazel_rules-0394a3b108412b8e543fd90255daa416e988c4a1",
    urls = [
        "https://mirror.bazel.build/github.com/drigz/bazel_rules/archive/0394a3b108412b8e543fd90255daa416e988c4a1.tar.gz",
        "https://github.com/drigz/bazel_rules/archive/0394a3b108412b8e543fd90255daa416e988c4a1.tar.gz",
    ],
)

# Point Cloud Library (PCL)
new_http_archive(
    name = "com_github_pointcloudlibrary_pcl",
    build_file = "//third_party:pcl.BUILD",
    sha256 = "5a102a2fbe2ba77c775bf92c4a5d2e3d8170be53a68c3a76cfc72434ff7b9783",
    strip_prefix = "pcl-pcl-1.8.1",
    urls = [
        "https://mirror.bazel.build/github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.tar.gz",
        "https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.tar.gz",
    ],
)

# FLANN
new_http_archive(
    name = "flann",
    build_file = "//third_party:flann.BUILD",
    strip_prefix = "flann-1.8.4-src",
    urls = [
        "https://www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip",
    ],
)

# HDF5
new_http_archive(
    name = "hdf5",
    url = "https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.1.tar.gz",
    strip_prefix = "hdf5-1.10.1",
    build_file = "third_party/hdf5.BUILD",
)

# Boost
# http_archive(
#     name = "com_github_nelhage_boost",
#     sha256 = "5c88fc077f6b8111e997fec5146e5f9940ae9a2016eb9949447fcb4b482bcdb3",
#     strip_prefix = "rules_boost-7289bb1d8f938fdf98078297768c122ee9e11c9e",
#     urls = [
#         "https://mirror.bazel.build/github.com/nelhage/rules_boost/archive/7289bb1d8f938fdf98078297768c122ee9e11c9e.tar.gz",
#         "https://github.com/nelhage/rules_boost/archive/7289bb1d8f938fdf98078297768c122ee9e11c9e.tar.gz",
#     ],
# )
#
# load("@com_github_nelhage_boost//:boost/boost.bzl", "boost_deps")
# boost_deps()

git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "239ce40e42ab0e3fe7ce84c2e9303ff8a277c41a",
    remote = "https://github.com/nelhage/rules_boost",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# Eigen
# Based on https://github.com/tensorflow/tensorflow/blob/master/third_party/eigen.BUILD
new_http_archive(
    name = "eigen_repo",
    build_file = "//third_party:eigen.BUILD",
    sha256 = "ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4",
    strip_prefix = "eigen-eigen-f3a22f35b044",
    urls = [
        "http://mirror.bazel.build/bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
        "https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz",
    ],
)
