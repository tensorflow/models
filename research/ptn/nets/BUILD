package(default_visibility = ["//visibility:public"])

py_library(
    name = "deeprotator_factory",
    srcs = ["deeprotator_factory.py"],
    deps = [
        ":ptn_encoder",
        ":ptn_im_decoder",
        ":ptn_rotator",
    ],
)

py_library(
    name = "im2vox_factory",
    srcs = ["im2vox_factory.py"],
    deps = [
        ":perspective_projector",
        ":ptn_encoder",
        ":ptn_vox_decoder",
    ],
)

py_library(
    name = "perspective_projector",
    srcs = ["perspective_projector.py"],
    deps = [
        ":perspective_transform",
    ],
)

py_library(
    name = "perspective_transform",
    srcs = ["perspective_transform.py"],
    deps = [
    ],
)

py_library(
    name = "ptn_encoder",
    srcs = ["ptn_encoder.py"],
    deps = [
    ],
)

py_library(
    name = "ptn_im_decoder",
    srcs = ["ptn_im_decoder.py"],
    deps = [
    ],
)

py_library(
    name = "ptn_rotator",
    srcs = ["ptn_rotator.py"],
    deps = [
    ],
)

py_library(
    name = "ptn_vox_decoder",
    srcs = ["ptn_vox_decoder.py"],
    deps = [
    ],
)
