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
