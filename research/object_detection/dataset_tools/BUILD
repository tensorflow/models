# Tensorflow Object Detection API: main runnables.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_binary(
    name = "create_kitti_tf_record",
    srcs = [
        "create_kitti_tf_record.py",
    ],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/lxml",
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:dataset_util",
        "//tensorflow_models/object_detection/utils:label_map_util",
        "//tensorflow_models/object_detection/utils:np_box_ops",
    ],
)

py_test(
    name = "create_kitti_tf_record_test",
    srcs = [
        "create_kitti_tf_record_test.py",
    ],
    deps = [
        ":create_kitti_tf_record",
        "//tensorflow",
    ],
)

py_binary(
    name = "create_pascal_tf_record",
    srcs = [
        "create_pascal_tf_record.py",
    ],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/lxml",
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:dataset_util",
        "//tensorflow_models/object_detection/utils:label_map_util",
    ],
)

py_test(
    name = "create_pascal_tf_record_test",
    srcs = [
        "create_pascal_tf_record_test.py",
    ],
    deps = [
        ":create_pascal_tf_record",
        "//tensorflow",
    ],
)

py_binary(
    name = "create_pet_tf_record",
    srcs = [
        "create_pet_tf_record.py",
    ],
    deps = [
        "//third_party/py/PIL:pil",
        "//third_party/py/lxml",
        "//tensorflow",
        "//tensorflow_models/object_detection/utils:dataset_util",
        "//tensorflow_models/object_detection/utils:label_map_util",
    ],
)

py_library(
    name = "oid_tfrecord_creation",
    srcs = ["oid_tfrecord_creation.py"],
    deps = [
        "//tensorflow",
        "//tensorflow_models/object_detection/core:standard_fields",
        "//tensorflow_models/object_detection/utils:dataset_util",
    ],
)
