# Description:
#   The Point Cloud Library (PCL) is a standalone, large scale, open project
#   for 2D/3D image and point cloud processing.

load("@bazel_rules//:config.bzl", "cc_fix_config")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # BSD

# TODO(rodrigoq): extract version number etc
cc_fix_config(
    name = "common_pcl_config",
    cmake = True,
    files = {"pcl_config.h.in": "common/include/pcl/pcl_config.h"},
    values = {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "PCL_MAJOR_VERSION": "1",
        "PCL_MINOR_VERSION": "8",
        "PCL_REVISION_VERSION": "1",
        "PCL_DEV_VERSION": "1",
        "PCL_VERSION": "1.8.1-dev",
        "HAVE_OPENNI": "1",
        "HAVE_QHULL": "1",
        "HAVE_POSIX_MEMALIGN": "1",
        "HAVE_MM_MALLOC": "1",
        "HAVE_SSE4_2_EXTENSIONS": "1",
        "HAVE_SSE4_1_EXTENSIONS": "1",
        "HAVE_SSSE3_EXTENSIONS": "1",
        "HAVE_SSE3_EXTENSIONS": "1",
        "HAVE_SSE2_EXTENSIONS": "1",
        "HAVE_SSE_EXTENSIONS": "1",
        "HAVE_PNG": "1",
        "VERBOSITY_LEVEL_INFO": "1",
        "VTK_RENDERING_BACKEND_OPENGL_VERSION": "1",
    },
    visibility = ["//visibility:private"],
)

BOOST_TARGETS = [
    "@boost//:algorithm",
    "@boost//:align",
    "@boost//:any",
    "@boost//:archive",
    "@boost//:array",
    "@boost//:asio",
    "@boost//:assert",
    "@boost//:atomic",
    "@boost//:beast",
    "@boost//:bimap",
    "@boost//:bind",
    "@boost//:call_traits",
    "@boost//:callable_traits",
    "@boost//:cerrno",
    "@boost//:checked_delete",
    "@boost//:chrono",
    "@boost//:circular_buffer",
    # "@boost//:compute",
    "@boost//:concept",
    "@boost//:concept_archetype",
    "@boost//:concept_check",
    "@boost//:config",
    "@boost//:container",
    # "@boost//:context",
    "@boost//:conversion",
    "@boost//:core",
    # "@boost//:coroutine",
    "@boost//:cstdint",
    "@boost//:current_function",
    "@boost//:date_time",
    "@boost//:detail",
    "@boost//:dynamic_bitset",
    "@boost//:enable_shared_from_this",
    # "@boost//:endian",
    "@boost//:exception",
    "@boost//:exception_ptr",
    # "@boost//:fiber",
    "@boost//:filesystem",
    "@boost//:foreach",
    "@boost//:format",
    "@boost//:function",
    "@boost//:function_types",
    "@boost//:functional",
    "@boost//:fusion",
    "@boost//:get_pointer",
    "@boost//:heap",
    # "@boost//:icl",
    "@boost//:integer",
    "@boost//:interprocess",
    "@boost//:intrusive",
    "@boost//:intrusive_ptr",
    "@boost//:io",
    "@boost//:iostreams",
    "@boost//:is_placeholder",
    "@boost//:iterator",
    "@boost//:lexical_cast",
    "@boost//:limits",
    "@boost//:math",
    "@boost//:mem_fn",
    "@boost//:move",
    "@boost//:mp11",
    "@boost//:mpl",
    "@boost//:multi_index",
    # "@boost//:multiprecision",
    "@boost//:noncopyable",
    "@boost//:none",
    "@boost//:numeric",
    "@boost//:numeric_conversion",
    "@boost//:numeric_ublas",
    "@boost//:operators",
    "@boost//:optional",
    "@boost//:parameter",
    # "@boost//:pool",
    "@boost//:predef",
    "@boost//:preprocessor",
    "@boost//:process",
    "@boost//:program_options",
    "@boost//:property_tree",
    "@boost//:ptr_container",
    "@boost//:random",
    "@boost//:range",
    "@boost//:ratio",
    # "@boost//:rational",
    "@boost//:ref",
    "@boost//:regex",
    "@boost//:scope_exit",
    "@boost//:scoped_array",
    "@boost//:scoped_ptr",
    "@boost//:serialization",
    "@boost//:shared_array",
    "@boost//:shared_ptr",
    "@boost//:signals2",
    "@boost//:smart_ptr",
    "@boost//:spirit",
    "@boost//:static_assert",
    "@boost//:swap",
    "@boost//:system",
    "@boost//:thread",
    "@boost//:throw_exception",
    "@boost//:timer",
    "@boost//:tokenizer",
    # "@boost//:tribool",
    "@boost//:tuple",
    "@boost//:type",
    "@boost//:type_index",
    "@boost//:type_traits",
    "@boost//:typeof",
    "@boost//:unordered",
    "@boost//:utility",
    "@boost//:uuid",
    "@boost//:variant",
    "@boost//:version",
    "@boost//:visit_each",
]

cc_library(
    name = "common",
    srcs = glob([
        "common/src/**/*.cpp",
    ]),
    hdrs = glob(
        [
            "common/include/pcl/*.h",
            "common/include/pcl/common/*.h",
            "common/include/pcl/common/fft/*.h",
            "common/include/pcl/console/*.h",
            "common/include/pcl/range_image/*.h",
            "common/include/pcl/ros/*.h",
        ],
        exclude = [
            "common/include/pcl/pcl_tests.h",
        ],
    ) + [
        "common/include/pcl/pcl_config.h",
    ],
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-unknown-pragmas",
        "-Wno-error=unknown-pragmas",
        "-Wno-comment",
    ],
    includes = [
        "common/include",
    ],
    textual_hdrs = glob(["common/include/**/impl/**/*.h*"]),
    deps = BOOST_TARGETS + [
        # "//base",
        "@eigen_repo//:eigen",
    ],
)

cc_library(
    name = "features",
    srcs = glob(["features/src/**/*.cpp"]),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-unused-variable",
        "-Wno-unknown-pragmas",
        "-Wno-error=unknown-pragmas",
    ],
    includes = [
        "features/include",
    ],
    textual_hdrs = glob(["features/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        "@eigen_repo//:eigen",
        "@hdf5//:hdf5",
        ":2d",
        ":common",
        ":kdtree",
        ":octree",
        ":search",
    ],
)

cc_library(
    name = "filters",
    srcs = glob(["filters/src/**/*.cpp"]),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-overloaded-virtual",
        "-Wno-string-conversion",
    ],
    includes = [
        "filters/include",
    ],
    textual_hdrs = glob(["filters/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        "@eigen_repo//:eigen",
        "@hdf5//:hdf5",
        ":common",
        ":kdtree",
        ":sample_consensus",
        ":search",
    ],
)

cc_library(
    name = "geometry",
    includes = [
        "geometry/include",
    ],
    textual_hdrs = glob(["geometry/include/**/*.h*"]),
    deps = BOOST_TARGETS + [":common"],
)

cc_library(
    name = "io",
    srcs = glob(["io/src/**/*.cpp"]),
    hdrs = glob(["io/include/**/*.h*"]),
    copts = [
        "$(STACK_FRAME_UNLIMITED)",
        "-Wno-implicit-fallthrough",
        "-fexceptions",
    ],
    includes = [
        "io/include",
    ],
    deps = BOOST_TARGETS + [
        # "//base",
        ":common",
    ],
)

cc_library(
    name = "kdtree",
    srcs = glob(["kdtree/src/**/*.cpp"]),
    hdrs = glob(["kdtree/include/**/*.h*"]),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
    ],
    includes = [
        "kdtree/include",
    ],
    deps = BOOST_TARGETS + [
        "@flann//:flann",
        ":common",
    ],
)

cc_library(
    name = "keypoints",
    srcs = glob(["keypoints/src/**/*.cpp"]),
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
    ] + select({
        ":opt": ["-Wno-unknown-pragmas"],
        "//conditions:default": ["-Wno-implicit-fallthrough"],
    }),
    includes = [
        "keypoints/include",
    ],
    textual_hdrs = glob(["keypoints/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        ":common",
        ":features",
        ":filters",
        ":kdtree",
        ":search",
    ],
)

cc_library(
    name = "octree",
    srcs = glob(["octree/src/**/*.cpp"]),
    copts = [
        "-Wno-overloaded-virtual",
    ],
    includes = [
        "octree/include",
    ],
    textual_hdrs = glob(["octree/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        ":common",
    ],
)

cc_library(
    name = "registration",
    srcs = glob(
        ["registration/src/**/*.cpp"],
        exclude = [
            "registration/src/pairwise_graph_registration.cpp",
        ],
    ),
    hdrs = glob(["registration/include/**/*.h*"]),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-unused-variable",
    ],
    includes = [
        "registration/include",
    ],
    deps = BOOST_TARGETS + [
        ":features",
        ":filters",
        ":sample_consensus",
    ],
)

cc_library(
    name = "sample_consensus",
    srcs = glob(["sample_consensus/src/**/*.cpp"]),
    hdrs = glob(["sample_consensus/include/**/*.h*"]),
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    includes = [
        "sample_consensus/include",
    ],
    deps = BOOST_TARGETS + [
        ":search",
    ],
)

cc_library(
    name = "search",
    srcs = glob(["search/src/**/*.cpp"]),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-overloaded-virtual",
        "-Wno-string-conversion",
    ],
    includes = [
        "search/include",
    ],
    textual_hdrs = glob(["search/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        "@eigen_repo//:eigen",
        ":common",
        ":kdtree",
        ":octree",
    ],
)

cc_library(
    name = "segmentation",
    srcs = glob(["segmentation/src/**/*.cpp"]),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-unused-variable",
    ],
    includes = [
        "segmentation/include",
    ],
    textual_hdrs = glob(["segmentation/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        "@eigen_repo//:eigen",
        ":common",
        ":features",
        ":geometry",
        ":kdtree",
        ":octree",
        ":sample_consensus",
        ":search",
    ],
)

cc_library(
    name = "2d",
    srcs = glob(
        ["2d/src/**/*.cpp"],
        exclude = ["2d/src/example*.cpp"],
    ),
    copts = [
        "-fexceptions",
        "-Wno-implicit-fallthrough",
        "-Wno-unused-variable",
    ],
    includes = [
        "2d/include",
    ],
    textual_hdrs = glob(["2d/include/**/*.h*"]),
    deps = BOOST_TARGETS + [
        "@eigen_repo//:eigen",
        ":common",
        ":filters",
    ],
)
