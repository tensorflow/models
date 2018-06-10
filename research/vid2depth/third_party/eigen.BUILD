# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.

licenses([
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

# Note: unsupported/Eigen is unsupported and might go away at any time.
EIGEN_FILES = [
    "Eigen/**",
    "unsupported/**",
]

# Files known to be under MPL2 license.
EIGEN_MPL2_HEADER_FILES = glob(
    EIGEN_FILES,
)

cc_library(
    name = "eigen",
    hdrs = EIGEN_MPL2_HEADER_FILES,
    defines = [
        "EIGEN_MPL2_ONLY",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
