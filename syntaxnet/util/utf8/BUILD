licenses(["notice"])

# Requires --copt -funsigned-char when compiling (unsigned chars).

cc_library(
    name = "unicodetext",
    srcs = [
        "unicodetext.cc",
        "unilib.cc",
    ],
    hdrs = [
        "unicodetext.h",
        "unilib.h",
        "unilib_utf8_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//syntaxnet:base",
        "//third_party/utf",
    ],
)

cc_test(
    name = "unicodetext_unittest",
    srcs = [
        "gtest_main.cc",
        "unicodetext_unittest.cc",
    ],
    deps = [
        "@tf//tensorflow/core:testlib",
        ":unicodetext",
    ],
)

cc_binary(
    name = "unicodetext_main",
    srcs = ["unicodetext_main.cc"],
    deps = [":unicodetext"],
)
