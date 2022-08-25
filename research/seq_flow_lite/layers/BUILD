py_strict_library = py_library

licenses(["notice"])

package(
    default_visibility = ["//:friends"],  # sequence projection
)

py_strict_library(
    name = "base_layers",
    srcs = ["base_layers.py"],
    srcs_version = "PY3",
    deps = [
        # package tensorflow
    ],
)

py_strict_library(
    name = "quantization_layers",
    srcs = ["quantization_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layers",
        # package tensorflow
    ],
)

py_strict_library(
    name = "normalization_layers",
    srcs = ["normalization_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layers",
        ":quantization_layers",
        # package tensorflow
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_strict_library(
    name = "dense_layers",
    srcs = ["dense_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layers",
        ":normalization_layers",
        ":quantization_layers",
        # package tensorflow
    ],
)

py_strict_library(
    name = "conv_layers",
    srcs = ["conv_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layers",
        ":normalization_layers",
        ":quantization_layers",
        # package tensorflow
    ],
)

py_strict_library(
    name = "projection_layers",
    srcs = ["projection_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layers",
        # package absl/logging
        # package tensorflow
        # "//tf_ops:sequence_string_projection_op"  # sequence projection
        "//tf_ops:sequence_string_projection_op_py",  # sequence projection
        # "//tf_ops:sequence_string_projection_op_v2"  # sequence projection
        "//tf_ops:sequence_string_projection_op_v2_py",  # sequence projection
    ],
)

py_strict_library(
    name = "misc_layers",
    srcs = ["misc_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":embedding_layers",
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:conv_layers",  # sequence projection
        "//layers:dense_layers",  # sequence projection
        "//layers:normalization_layers",  # sequence projection
        "//layers:quantization_layers",  # sequence projection
    ],
)

py_strict_library(
    name = "qrnn_layers",
    srcs = ["qrnn_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layers",
        ":conv_layers",
        ":dense_layers",
        ":quantization_layers",
        # package tensorflow
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_strict_library(
    name = "embedding_layers",
    srcs = ["embedding_layers.py"],
    srcs_version = "PY3",
    deps = [
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:quantization_layers",  # sequence projection
    ],
)

py_strict_library(
    name = "transformer_layers",
    srcs = ["transformer_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":embedding_layers",
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:dense_layers",  # sequence projection
        "//layers:normalization_layers",  # sequence projection
        "//layers:quantization_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)
