new_http_archive(
  name = "gmock_archive",
  url = "https://googlemock.googlecode.com/files/gmock-1.7.0.zip",
  sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
  build_file = "tensorflow/google/protobuf/gmock.BUILD",
)

new_http_archive(
  name = "eigen_archive",
  url = "https://bitbucket.org/eigen/eigen/get/70505a059011.tar.gz",
  sha256 = "9751bd3485a9b373bc1b40626feac37484099e54b2b47a93d3da8bf1312a7beb",
  build_file = "tensorflow/eigen.BUILD",
)

bind(
  name = "gtest",
  actual = "@gmock_archive//:gtest",
)

bind(
  name = "gtest_main",
  actual = "@gmock_archive//:gtest_main",
)

local_repository(
    name = "tf",
    path = __workspace_dir__ + "/tensorflow",
)

# ===== gRPC dependencies =====

bind(
    name = "libssl",
    actual = "@boringssl_git//:ssl",
)

git_repository(
    name = "boringssl_git",
    commit = "436432d849b83ab90f18773e4ae1c7a8f148f48d",
    init_submodules = True,
    remote = "https://github.com/mdsteele/boringssl-bazel.git",
)

bind(
    name = "protobuf_clib",
    actual = "@tf//google/protobuf:protobuf",
)

bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)

new_http_archive(
    name = "zlib_archive",
    build_file = "zlib.BUILD",
    sha256 = "879d73d8cd4d155f31c1f04838ecd567d34bebda780156f0e82a20721b3973d5",
    strip_prefix = "zlib-1.2.8",
    url = "http://zlib.net/zlib128.zip",
)

# =========================

git_repository(
  name = "re2",
  remote = "https://github.com/google/re2.git",
  commit = "791beff",
)

new_http_archive(
  name = "jpeg_archive",
  url = "http://www.ijg.org/files/jpegsrc.v9a.tar.gz",
  sha256 = "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7",
  build_file = "tensorflow/jpeg.BUILD",
)

new_http_archive(
  name = "png_archive",
  url = "https://storage.googleapis.com/libpng-public-archive/libpng-1.2.53.tar.gz",
  sha256 = "e05c9056d7f323088fd7824d8c6acc03a4a758c4b4916715924edc5dd3223a72",
  build_file = "tensorflow/png.BUILD",
)

new_http_archive(
  name = "six_archive",
  url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
  sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
  build_file = "tensorflow/six.BUILD",
)

bind(
  name = "six",
  actual = "@six_archive//:six",
)
