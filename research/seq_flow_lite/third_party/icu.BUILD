licenses(["notice"])
exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "icu4c",
    srcs = glob(
        [
            "icu4c/source/common/*.c",
            "icu4c/source/common/*.cpp",
            "icu4c/source/stubdata/*.cpp",
        ],
    ),
    hdrs = glob([
        "icu4c/source/common/*.h",
    ]) + glob([
        "icu4c/source/common/unicode/*.h",
    ]),
    copts = [
        "-DU_COMMON_IMPLEMENTATION",
        "-Wno-deprecated-declarations",
    ],
    linkopts = [
        "-ldl",
    ],
)
