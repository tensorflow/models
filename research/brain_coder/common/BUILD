licenses(["notice"])

package(default_visibility = [
    "//:__subpackages__",
])

py_library(
    name = "bf",
    srcs = ["bf.py"],
)

py_test(
    name = "bf_test",
    srcs = ["bf_test.py"],
    deps = [
        ":bf",
        # tensorflow dep
    ],
)

py_library(
    name = "config_lib",
    srcs = ["config_lib.py"],
)

py_test(
    name = "config_lib_test",
    srcs = ["config_lib_test.py"],
    deps = [
        ":config_lib",
        # tensorflow dep
    ],
)

py_library(
    name = "reward",
    srcs = ["reward.py"],
)

py_test(
    name = "reward_test",
    srcs = ["reward_test.py"],
    deps = [
        ":reward",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "rollout",
    srcs = ["rollout.py"],
    deps = [
        ":utils",
        # numpy dep
        # scipy dep
    ],
)

py_test(
    name = "rollout_test",
    srcs = ["rollout_test.py"],
    deps = [
        ":rollout",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "schedules",
    srcs = ["schedules.py"],
    deps = [":config_lib"],
)

py_test(
    name = "schedules_test",
    srcs = ["schedules_test.py"],
    deps = [
        ":config_lib",
        ":schedules",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # file dep
        # absl dep /logging
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        # numpy dep
        # tensorflow dep
    ],
)
