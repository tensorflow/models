# Tensorflow Object Detection API: dataset tools.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

# Apache 2.0

py_binary(
    name = "create_coco_tf_record",
    srcs = [
        "create_coco_tf_record.py",
    ],
    deps = [
        "//PIL:pil",
        "//pycocotools",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
    ],
)

py_test(
    name = "create_coco_tf_record_test",
    srcs = [
        "create_coco_tf_record_test.py",
    ],
    deps = [
        ":create_coco_tf_record",
        "//tensorflow",
    ],
)

py_binary(
    name = "create_kitti_tf_record",
    srcs = [
        "create_kitti_tf_record.py",
    ],
    deps = [
        "//PIL:pil",
        "//lxml",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
        "//tensorflow/models/research/object_detection/utils:np_box_ops",
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
        "//PIL:pil",
        "//lxml",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
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
        "//PIL:pil",
        "//lxml",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
    ],
)

py_library(
    name = "oid_tfrecord_creation",
    srcs = ["oid_tfrecord_creation.py"],
    deps = [
        "//tensorflow",
        "//tensorflow/models/research/object_detection/core:standard_fields",
        "//tensorflow/models/research/object_detection/utils:dataset_util",
    ],
)

py_test(
    name = "oid_tfrecord_creation_test",
    srcs = ["oid_tfrecord_creation_test.py"],
    deps = [
        ":oid_tfrecord_creation",
        "//contextlib2",
        "//pandas",
        "//tensorflow",
    ],
)

py_binary(
    name = "create_oid_tf_record",
    srcs = ["create_oid_tf_record.py"],
    deps = [
        ":oid_tfrecord_creation",
        "//contextlib2",
        "//pandas",
        "//tensorflow",
        "//tensorflow/models/research/object_detection/utils:label_map_util",
    ],
)
