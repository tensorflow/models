# Domain Separation Networks

package(
    default_visibility = [
        ":internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//domain_adaptation/...",
    ],
)

py_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "losses",
    srcs = [
        "losses.py",
    ],
    deps = [
        ":grl_op_grads_py",
        ":grl_op_shapes_py",
        ":grl_ops",
        ":utils",
    ],
)

py_test(
    name = "losses_test",
    srcs = [
        "losses_test.py",
    ],
    deps = [
        ":losses",
        ":utils",
    ],
)

py_library(
    name = "dsn",
    srcs = [
        "dsn.py",
    ],
    deps = [
        ":grl_op_grads_py",
        ":grl_op_shapes_py",
        ":grl_ops",
        ":losses",
        ":models",
        ":utils",
    ],
)

py_test(
    name = "dsn_test",
    srcs = [
        "dsn_test.py",
    ],
    deps = [
        ":dsn",
    ],
)

py_binary(
    name = "dsn_train",
    srcs = [
        "dsn_train.py",
    ],
    deps = [
        ":dsn",
        ":models",
        "//domain_adaptation/datasets:dataset_factory",
    ],
)

py_binary(
    name = "dsn_eval",
    srcs = [
        "dsn_eval.py",
    ],
    deps = [
        ":dsn",
        ":models",
        "//domain_adaptation/datasets:dataset_factory",
    ],
)

py_test(
    name = "models_test",
    srcs = [
        "models_test.py",
    ],
    deps = [
        ":models",
        "//domain_adaptation/datasets:dataset_factory",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    deps = [
    ],
)

py_library(
    name = "grl_op_grads_py",
    srcs = [
        "grl_op_grads.py",
    ],
    deps = [
        ":grl_ops",
    ],
)

py_library(
    name = "grl_op_shapes_py",
    srcs = [
        "grl_op_shapes.py",
    ],
    deps = [
    ],
)

py_library(
    name = "grl_ops",
    srcs = ["grl_ops.py"],
    data = ["_grl_ops.so"],
)

py_test(
    name = "grl_ops_test",
    size = "small",
    srcs = ["grl_ops_test.py"],
    deps = [
        ":grl_op_grads_py",
        ":grl_op_shapes_py",
        ":grl_ops",
    ],
)
