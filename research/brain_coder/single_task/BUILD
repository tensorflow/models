licenses(["notice"])

package(default_visibility = [
    "//learning/brain/research/neural_coder:__subpackages__",
])

load("@subpar//:subpar.bzl", "par_binary")

par_binary(
    name = "run",
    srcs = ["run.py"],
    deps = [
        ":defaults",
        ":ga_train",
        ":pg_train",
        # absl dep :app
        # absl dep /flags
        # absl dep /logging
    ],
)

par_binary(
    name = "tune",
    srcs = ["tune.py"],
    deps = [
        ":defaults",
        ":run",
        # file dep
        # absl dep :app
        # absl dep /flags
        # absl dep /logging
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "ga_train",
    srcs = ["ga_train.py"],
    deps = [
        ":data",
        ":defaults",
        ":ga_lib",
        ":results_lib",
        # file dep
        # absl dep /flags
        # absl dep /logging
        # numpy dep
        # tensorflow dep
        "//common:utils",  # project
    ],
)

py_library(
    name = "ga_lib",
    srcs = ["ga_lib.py"],
    deps = [
        ":misc",
        # absl dep /flags
        # absl dep /logging
        # numpy dep
        "//common:bf",  # project
        "//common:utils",  # project
    ],
)

py_test(
    name = "ga_train_test",
    srcs = ["ga_train_test.py"],
    deps = [
        ":defaults",
        ":run",
        # absl dep /flags
        # tensorflow dep
    ],
)

py_library(
    name = "pg_train",
    srcs = ["pg_train.py"],
    deps = [
        ":data",
        ":defaults",
        ":pg_agent",
        ":results_lib",
        # file dep
        # absl dep /flags
        # absl dep /logging
        # tensorflow dep
        # tensorflow internal dep  # build_cleaner: keep
    ],
)

py_library(
    name = "pg_agent",
    srcs = ["pg_agent.py"],
    deps = [
        ":misc",
        # file dep
        # absl dep /logging
        # numpy dep
        # tensorflow dep
        "//common:rollout",  # project
        "//common:utils",  # project
    ],
)

py_test(
    name = "pg_agent_test",
    srcs = ["pg_agent_test.py"],
    deps = [
        ":data",
        ":defaults",
        ":misc",
        ":pg_agent",
        ":pg_train",
        # absl dep /logging
        # numpy dep
        # tensorflow dep
        "//common:utils",  # project
    ],
)

py_library(
    name = "defaults",
    srcs = ["defaults.py"],
    deps = [
        # absl dep /logging
        "//common:config_lib",  # project
    ],
)

py_library(
    name = "misc",
    srcs = ["misc.py"],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        ":code_tasks",
        # absl dep /logging
    ],
)

py_library(
    name = "code_tasks",
    srcs = ["code_tasks.py"],
    deps = [
        ":misc",
        ":test_tasks",
        # absl dep /logging
        # numpy dep
        "//common:bf",  # project
        "//common:reward",  # project
    ],
)

py_test(
    name = "code_tasks_test",
    srcs = ["code_tasks_test.py"],
    deps = [
        ":code_tasks",
        ":defaults",
        # numpy dep
        # tensorflow dep
    ],
)

py_library(
    name = "test_tasks",
    srcs = ["test_tasks.py"],
    deps = [
        ":misc",
        "//common:reward",  # project
    ],
)

py_test(
    name = "test_tasks_test",
    srcs = ["test_tasks_test.py"],
    deps = [
        ":misc",
        ":test_tasks",
        # numpy dep
        # tensorflow dep
    ],
)

py_test(
    name = "pg_train_test",
    size = "large",
    srcs = ["pg_train_test.py"],
    deps = [
        ":defaults",
        ":run",
        # absl dep /logging
        # tensorflow dep
    ],
)

py_library(
    name = "results_lib",
    srcs = ["results_lib.py"],
    deps = [
        # file dep
        # tensorflow dep
    ],
)

py_test(
    name = "results_lib_test",
    srcs = ["results_lib_test.py"],
    deps = [
        ":results_lib",
        # tensorflow dep
    ],
)

par_binary(
    name = "aggregate_experiment_results",
    srcs = ["aggregate_experiment_results.py"],
    deps = [
        ":misc",
        ":results_lib",
        # file dep
        # absl dep :app
        # absl dep /flags
        # numpy dep
        # tensorflow dep
    ],
)

par_binary(
    name = "aggregate_tuning_results",
    srcs = ["aggregate_tuning_results.py"],
    deps = [
        # file dep
        # absl dep :app
        # absl dep /flags
        # tensorflow dep
    ],
)
