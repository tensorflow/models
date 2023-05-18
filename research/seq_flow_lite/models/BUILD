licenses(["notice"])

package(default_visibility = ["//:friends"])  # sequence projection

py_library(
    name = "prado",
    srcs = ["prado.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:conv_layers",  # sequence projection
        "//layers:dense_layers",  # sequence projection
        "//layers:projection_layers",  # sequence projection
        "//layers:quantization_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_library(
    name = "pqrnn",
    srcs = ["pqrnn.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:dense_layers",  # sequence projection
        "//layers:misc_layers",  # sequence projection
        "//layers:projection_layers",  # sequence projection
        "//layers:qrnn_layers",  # sequence projection
        "//layers:quantization_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
    ],
)

py_library(
    name = "byteqrnn",
    srcs = ["byteqrnn.py"],
    srcs_version = "PY3",
    deps = [
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:dense_layers",  # sequence projection
        "//layers:embedding_layers",  # sequence projection
        "//layers:misc_layers",  # sequence projection
        "//layers:qrnn_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_library(
    name = "charformer",
    srcs = ["charformer.py"],
    srcs_version = "PY3",
    deps = [
        ":transformer_encoder",
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:embedding_layers",  # sequence projection
        "//layers:misc_layers",  # sequence projection
        "//layers:normalization_layers",  # sequence projection
        "//layers:quantization_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_library(
    name = "transformer_encoder",
    srcs = ["transformer_encoder.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:embedding_layers",  # sequence projection
        "//layers:transformer_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_library(
    name = "transformer_uniform_attn_decoder",
    srcs = ["transformer_uniform_attn_decoder.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package tensor2tensor/utils:beam_search
        # package tensorflow
        "//layers:base_layers",  # sequence projection
        "//layers:embedding_layers",  # sequence projection
        "//layers:misc_layers",  # sequence projection
        "//layers:transformer_layers",  # sequence projection
        # "//tf_ops:tf_custom_ops"  # sequence projection
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)
