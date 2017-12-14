package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//tcn/...",
    ],
)

py_binary(
    name = "download_pretrained",
    srcs = [
        "download_pretrained.py",
    ],
)

py_binary(
    name = "generate_videos",
    srcs = [
        "generate_videos.py",
    ],
    main = "generate_videos.py",
    deps = [
        ":data_providers",
        ":get_estimator",
        ":util",
    ],
)

py_test(
    name = "svtcn_loss_test",
    size = "medium",
    srcs = [
        "estimators/svtcn_loss.py",
        "estimators/svtcn_loss_test.py",
    ],
    deps = [
        ":util",
    ],
)

py_library(
    name = "data_providers",
    srcs = [
        "data_providers.py",
    ],
    deps = [
        ":preprocessing",
    ],
)

py_test(
    name = "data_providers_test",
    size = "large",
    srcs = ["data_providers_test.py"],
    deps = [
        ":data_providers",
    ],
)

py_library(
    name = "preprocessing",
    srcs = [
        "preprocessing.py",
    ],
)

py_binary(
    name = "get_estimator",
    srcs = [
        "estimators/get_estimator.py",
    ],
    deps = [
        ":mvtcn_estimator",
        ":svtcn_estimator",
    ],
)

py_binary(
    name = "base_estimator",
    srcs = [
        "estimators/base_estimator.py",
        "model.py",
    ],
    deps = [
        ":data_providers",
        ":util",
    ],
)

py_library(
    name = "util",
    srcs = [
        "utils/luatables.py",
        "utils/progress.py",
        "utils/util.py",
    ],
)

py_binary(
    name = "mvtcn_estimator",
    srcs = [
        "estimators/mvtcn_estimator.py",
    ],
    deps = [
        ":base_estimator",
    ],
)

py_binary(
    name = "svtcn_estimator",
    srcs = [
        "estimators/svtcn_estimator.py",
        "estimators/svtcn_loss.py",
    ],
    deps = [
        ":base_estimator",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    deps = [
        ":data_providers",
        ":get_estimator",
        ":util",
    ],
)

py_binary(
    name = "labeled_eval",
    srcs = [
        "labeled_eval.py",
    ],
    deps = [
        ":get_estimator",
    ],
)

py_test(
    name = "labeled_eval_test",
    size = "small",
    srcs = ["labeled_eval_test.py"],
    deps = [
        ":labeled_eval",
    ],
)

py_binary(
    name = "eval",
    srcs = [
        "eval.py",
    ],
    deps = [
        ":get_estimator",
    ],
)

py_binary(
    name = "alignment",
    srcs = [
        "alignment.py",
    ],
    deps = [
        ":get_estimator",
    ],
)

py_binary(
    name = "visualize_embeddings",
    srcs = [
        "visualize_embeddings.py",
    ],
    deps = [
        ":data_providers",
        ":get_estimator",
        ":util",
    ],
)

py_binary(
    name = "webcam",
    srcs = [
        "dataset/webcam.py",
    ],
    main = "dataset/webcam.py",
)

py_binary(
    name = "images_to_videos",
    srcs = [
        "dataset/images_to_videos.py",
    ],
    main = "dataset/images_to_videos.py",
)

py_binary(
    name = "videos_to_tfrecords",
    srcs = [
        "dataset/videos_to_tfrecords.py",
    ],
    main = "dataset/videos_to_tfrecords.py",
    deps = [
        ":preprocessing",
    ],
)
