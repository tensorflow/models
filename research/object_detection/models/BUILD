# Tensorflow Object Detection API: Models.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_library(
    name = "feature_map_generators",
    srcs = [
        "feature_map_generators.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:ops",
    ],
)

py_test(
    name = "feature_map_generators_test",
    srcs = [
        "feature_map_generators_test.py",
    ],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
    ],
)

py_library(
    name = "ssd_feature_extractor_test",
    srcs = [
        "ssd_feature_extractor_test.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:test_case",
    ],
)

py_library(
    name = "ssd_inception_v2_feature_extractor",
    srcs = [
        "ssd_inception_v2_feature_extractor.py",
    ],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:ssd_meta_arch",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:shape_utils",
        "//third_party/tensorflow_models/slim:inception_v2",
    ],
)

py_library(
    name = "ssd_inception_v3_feature_extractor",
    srcs = [
        "ssd_inception_v3_feature_extractor.py",
    ],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:ssd_meta_arch",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:shape_utils",
        "//third_party/tensorflow_models/slim:inception_v3",
    ],
)

py_library(
    name = "ssd_mobilenet_v1_feature_extractor",
    srcs = ["ssd_mobilenet_v1_feature_extractor.py"],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:ssd_meta_arch",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:shape_utils",
        "//third_party/tensorflow_models/slim:mobilenet_v1",
    ],
)

py_library(
    name = "embedded_ssd_mobilenet_v1_feature_extractor",
    srcs = ["embedded_ssd_mobilenet_v1_feature_extractor.py"],
    deps = [
        ":feature_map_generators",
        ":ssd_mobilenet_v1_feature_extractor",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//third_party/tensorflow_models/slim:mobilenet_v1",
    ],
)

py_library(
    name = "ssd_resnet_v1_fpn_feature_extractor",
    srcs = ["ssd_resnet_v1_fpn_feature_extractor.py"],
    deps = [
        ":feature_map_generators",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:ssd_meta_arch",
        "//tensorflow/models/research/object_detection/utils:ops",
        "//tensorflow/models/research/object_detection/utils:shape_utils",
        "//third_party/tensorflow_models/slim:resnet_v1",
    ],
)

py_library(
    name = "ssd_resnet_v1_fpn_feature_extractor_testbase",
    srcs = ["ssd_resnet_v1_fpn_feature_extractor_testbase.py"],
    deps = [
        "//tensorflow/models/research/object_detection/models:ssd_feature_extractor_test",
    ],
)

py_test(
    name = "ssd_resnet_v1_fpn_feature_extractor_test",
    timeout = "long",
    srcs = ["ssd_resnet_v1_fpn_feature_extractor_test.py"],
    deps = [
        ":ssd_resnet_v1_fpn_feature_extractor",
        ":ssd_resnet_v1_fpn_feature_extractor_testbase",
        "//tensorflow",
    ],
)

py_test(
    name = "ssd_inception_v2_feature_extractor_test",
    srcs = [
        "ssd_inception_v2_feature_extractor_test.py",
    ],
    deps = [
        ":ssd_feature_extractor_test",
        ":ssd_inception_v2_feature_extractor",
        "//tensorflow",
    ],
)

py_test(
    name = "ssd_inception_v3_feature_extractor_test",
    srcs = [
        "ssd_inception_v3_feature_extractor_test.py",
    ],
    deps = [
        ":ssd_feature_extractor_test",
        ":ssd_inception_v3_feature_extractor",
        "//tensorflow",
    ],
)

py_test(
    name = "ssd_mobilenet_v1_feature_extractor_test",
    srcs = ["ssd_mobilenet_v1_feature_extractor_test.py"],
    deps = [
        ":ssd_feature_extractor_test",
        ":ssd_mobilenet_v1_feature_extractor",
        "//tensorflow",
    ],
)

py_test(
    name = "embedded_ssd_mobilenet_v1_feature_extractor_test",
    srcs = ["embedded_ssd_mobilenet_v1_feature_extractor_test.py"],
    deps = [
        ":embedded_ssd_mobilenet_v1_feature_extractor",
        ":ssd_feature_extractor_test",
        "//tensorflow",
    ],
)

py_test(
    name = "faster_rcnn_nas_feature_extractor_test",
    srcs = [
        "faster_rcnn_nas_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_nas_feature_extractor",
        "//tensorflow",
    ],
)

py_library(
    name = "faster_rcnn_nas_feature_extractor",
    srcs = [
        "faster_rcnn_nas_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//third_party/tensorflow_models/slim:nasnet",
    ],
)

py_library(
    name = "faster_rcnn_inception_resnet_v2_feature_extractor",
    srcs = [
        "faster_rcnn_inception_resnet_v2_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//third_party/tensorflow_models/slim:inception_resnet_v2",
    ],
)

py_test(
    name = "faster_rcnn_inception_resnet_v2_feature_extractor_test",
    srcs = [
        "faster_rcnn_inception_resnet_v2_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_inception_resnet_v2_feature_extractor",
        "//tensorflow",
    ],
)

py_library(
    name = "faster_rcnn_inception_v2_feature_extractor",
    srcs = [
        "faster_rcnn_inception_v2_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//third_party/tensorflow_models/slim:inception_v2",
    ],
)

py_test(
    name = "faster_rcnn_inception_v2_feature_extractor_test",
    srcs = [
        "faster_rcnn_inception_v2_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_inception_v2_feature_extractor",
        "//tensorflow",
    ],
)

py_library(
    name = "faster_rcnn_resnet_v1_feature_extractor",
    srcs = [
        "faster_rcnn_resnet_v1_feature_extractor.py",
    ],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/meta_architectures:faster_rcnn_meta_arch",
        "//third_party/tensorflow_models/slim:resnet_utils",
        "//third_party/tensorflow_models/slim:resnet_v1",
    ],
)

py_test(
    name = "faster_rcnn_resnet_v1_feature_extractor_test",
    srcs = [
        "faster_rcnn_resnet_v1_feature_extractor_test.py",
    ],
    deps = [
        ":faster_rcnn_resnet_v1_feature_extractor",
        "//tensorflow",
    ],
)
