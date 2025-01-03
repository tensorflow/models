_CHECK_VERSION = """
PROTOC_VERSION=$$($(location @protobuf_protoc//:protoc_bin) --version \
  | cut -d' ' -f2 | sed -e 's/\\./ /g')
PROTOC_VERSION=$$(printf '%d%03d%03d' $${PROTOC_VERSION})
TF_PROTO_VERSION=$$(grep '#define PROTOBUF_MIN_PROTOC_VERSION' \
  $(location tf_includes/google/protobuf/port_def.inc) | cut -d' ' -f3)
if [ "$${PROTOC_VERSION}" -ne "$${TF_PROTO_VERSION}" ]; then
  echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 1>&2
  echo Your protoc version does not match the tensorflow proto header \
       required version: "$${PROTOC_VERSION}" vs. "$${TF_PROTO_VERSION}" 1>&2
  echo Please update the PROTOC_VERSION in your WORKSPACE file. 1>&2
  echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!! 1>&2
  false
else
  touch $@
fi
"""

genrule(
    name = "compare_protobuf_version",
    outs = ["versions_compared"],
    srcs = [
        "tf_includes/google/protobuf/port_def.inc",
    ],
    tools = ["@protobuf_protoc//:protoc_bin"],
    cmd = _CHECK_VERSION,
)

cc_library(
    name = "includes",
    data = [":versions_compared"],
    hdrs = glob([
        "tf_includes/google/protobuf/*.h",
        "tf_includes/google/protobuf/*.inc",
        "tf_includes/google/protobuf/**/*.h",
        "tf_includes/google/protobuf/**/*.inc",
    ]),
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
