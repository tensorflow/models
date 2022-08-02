licenses(["notice"])

package(
    default_visibility = ["//:friends"],  # sequence projection
)

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
        "//tf_ops:tf_custom_ops_py",  # sequence projection
    ],
)

py_library(
    name = "byteqrnn",
    srcs = ["byteqrnn.py"],
    srcs_version = "PY3",
    deps = [
        # package tensorflow
        "//layers:base_layers",
        "//layers:dense_layers",
        "//layers:embedding_layers",
        "//layers:misc_layers",
        "//layers:qrnn_layers",
        # //tf_ops:tf_custom_ops",
        "//tf_ops:tf_custom_ops_py",
    ],
)

py_library(
    name = "charformer",
    srcs = ["charformer.py"],
    srcs_version = "PY3",
    deps = [
        ":transformer_encoder",
        # package tensorflow
        "//layers:base_layers",
        "//layers:embedding_layers",
        "//layers:misc_layers",
        "//layers:normalization_layers",
        "//layers:quantization_layers",
        # "//tf_ops:tf_custom_ops",
        "//tf_ops:tf_custom_ops_py",
    ],
)

py_library(
    name = "transformer_encoder",
    srcs = ["transformer_encoder.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package tensorflow
        "//layers:base_layers",
        "//layers:embedding_layers",
        "//layers:transformer_layers",
        # "//tf_ops:tf_custom_ops",
        "//tf_ops:tf_custom_ops_py",
    ],
)

py_library(
    name = "transformer_uniform_attn_decoder",
    srcs = ["transformer_uniform_attn_decoder.py"],
    srcs_version = "PY3",
    deps = [
        # package absl/logging
        # package tensorflow
        # tensor2tensor/utils:beam_search",
        "//layers:base_layers",
        "//layers:embedding_layers",
        "//layers:misc_layers",
        "//layers:transformer_layers",
        "//tf_ops:tf_custom_ops",
        "//tf_ops:tf_custom_ops_py",
    ],
)
