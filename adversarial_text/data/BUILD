licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//adversarial_text:__subpackages__",
    ],
)

py_binary(
    name = "gen_vocab",
    srcs = ["gen_vocab.py"],
    deps = [
        ":data_utils",
        ":document_generators",
        # tensorflow dep,
    ],
)

py_binary(
    name = "gen_data",
    srcs = ["gen_data.py"],
    deps = [
        ":data_utils",
        ":document_generators",
        # tensorflow dep,
    ],
)

py_library(
    name = "document_generators",
    srcs = ["document_generators.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "data_utils_test",
    srcs = ["data_utils_test.py"],
    deps = [
        ":data_utils",
        # tensorflow dep,
    ],
)
