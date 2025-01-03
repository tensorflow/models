# Description:
#   Contains files for loading, training and evaluating TF-Slim-based models.
# load("//devtools/python/blaze:python3.bzl", "py2and3_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "dataset_utils",
    srcs = ["datasets/dataset_utils.py"],
    deps = [
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

sh_binary(
    name = "download_and_convert_imagenet",
    srcs = ["datasets/download_and_convert_imagenet.sh"],
    data = [
        "datasets/download_imagenet.sh",
        "datasets/imagenet_2012_validation_synset_labels.txt",
        "datasets/imagenet_lsvrc_2015_synsets.txt",
        "datasets/imagenet_metadata.txt",
        "datasets/preprocess_imagenet_validation_data.py",
        "datasets/process_bounding_boxes.py",
        ":build_imagenet_data",
    ],
)

py_binary(
    name = "build_imagenet_data",
    srcs = ["datasets/build_imagenet_data.py"],
    python_version = "PY3",
    deps = [
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_cifar10",
    srcs = ["datasets/download_and_convert_cifar10.py"],
    deps = [
        ":dataset_utils",
        # "//numpy",
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_flowers",
    srcs = ["datasets/download_and_convert_flowers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_utils",
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_mnist",
    srcs = ["datasets/download_and_convert_mnist.py"],
    deps = [
        ":dataset_utils",
        # "//numpy",
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_visualwakewords_lib",
    srcs = ["datasets/download_and_convert_visualwakewords_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_utils",
        "//third_party/py/PIL:pil",
        "//third_party/py/contextlib2",
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_visualwakewords",
    srcs = ["datasets/download_and_convert_visualwakewords.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":download_and_convert_visualwakewords_lib",
        # "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_data",
    srcs = ["download_and_convert_data.py"],
    python_version = "PY3",
    deps = [
        ":download_and_convert_cifar10",
        ":download_and_convert_flowers",
        ":download_and_convert_mnist",
        ":download_and_convert_visualwakewords",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
    ],
)

py_library(
    name = "cifar10",
    srcs = ["datasets/cifar10.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "flowers",
    srcs = ["datasets/flowers.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "imagenet",
    srcs = ["datasets/imagenet.py"],
    deps = [
        ":dataset_utils",
        "//third_party/py/six",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "mnist",
    srcs = ["datasets/mnist.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "visualwakewords",
    srcs = ["datasets/visualwakewords.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "dataset_factory",
    srcs = ["datasets/dataset_factory.py"],
    deps = [
        ":cifar10",
        ":flowers",
        ":imagenet",
        ":mnist",
        ":visualwakewords",
    ],
)

py_library(
    name = "model_deploy",
    srcs = ["deployment/model_deploy.py"],
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "model_deploy_test",
    srcs = ["deployment/model_deploy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_deploy",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "cifarnet_preprocessing",
    srcs = ["preprocessing/cifarnet_preprocessing.py"],
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception_preprocessing",
    srcs = ["preprocessing/inception_preprocessing.py"],
    deps = [
        # "//tensorflow",
        # "//tensorflow/python:control_flow_ops",
    ],
)

py_library(
    name = "lenet_preprocessing",
    srcs = ["preprocessing/lenet_preprocessing.py"],
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "vgg_preprocessing",
    srcs = ["preprocessing/vgg_preprocessing.py"],
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "preprocessing_factory",
    srcs = ["preprocessing/preprocessing_factory.py"],
    deps = [
        ":cifarnet_preprocessing",
        ":inception_preprocessing",
        ":lenet_preprocessing",
        ":vgg_preprocessing",
        "//third_party/py/tf_slim:slim",
    ],
)

# Typical networks definitions.

py_library(
    name = "nets",
    deps = [
        ":alexnet",
        ":cifarnet",
        ":cyclegan",
        ":i3d",
        ":inception",
        ":lenet",
        ":mobilenet",
        ":nasnet",
        ":overfeat",
        ":pix2pix",
        ":pnasnet",
        ":resnet_v1",
        ":resnet_v2",
        ":s3dg",
        ":vgg",
    ],
)

py_library(
    name = "alexnet",
    srcs = ["nets/alexnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "alexnet_test",
    size = "medium",
    srcs = ["nets/alexnet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":alexnet",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "cifarnet",
    srcs = ["nets/cifarnet.py"],
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "cyclegan",
    srcs = ["nets/cyclegan.py"],
    deps = [
        # "//numpy",
        "//third_party/py/six",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/python:tensor_util",
    ],
)

py_test(  # py2and3_test
    name = "cyclegan_test",
    srcs = ["nets/cyclegan_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":cyclegan",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
    ],
)

py_library(
    name = "dcgan",
    srcs = ["nets/dcgan.py"],
    deps = [
        "//third_party/py/six",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "dcgan_test",
    srcs = ["nets/dcgan_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":dcgan",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "i3d",
    srcs = ["nets/i3d.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":i3d_utils",
        ":s3dg",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "i3d_test",
    size = "large",
    srcs = ["nets/i3d_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":i3d",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/six",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "i3d_utils",
    srcs = ["nets/i3d_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception",
    srcs = ["nets/inception.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_resnet_v2",
        ":inception_v1",
        ":inception_v2",
        ":inception_v3",
        ":inception_v4",
    ],
)

py_library(
    name = "inception_utils",
    srcs = ["nets/inception_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception_v1",
    srcs = ["nets/inception_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception_v2",
    srcs = ["nets/inception_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception_v3",
    srcs = ["nets/inception_v3.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception_v4",
    srcs = ["nets/inception_v4.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "inception_resnet_v2",
    srcs = ["nets/inception_resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "inception_v1_test",
    size = "large",
    srcs = ["nets/inception_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "inception_v2_test",
    size = "large",
    srcs = ["nets/inception_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "inception_v3_test",
    size = "large",
    srcs = ["nets/inception_v3_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "inception_v4_test",
    size = "large",
    srcs = ["nets/inception_v4_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "inception_resnet_v2_test",
    size = "large",
    srcs = ["nets/inception_resnet_v2_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "lenet",
    srcs = ["nets/lenet.py"],
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "mobilenet_v1",
    srcs = ["nets/mobilenet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "mobilenet_common",
    srcs = [
        "nets/mobilenet/conv_blocks.py",
        "nets/mobilenet/mobilenet.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = ["nets/mobilenet/mobilenet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_common",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "mobilenet_v3",
    srcs = ["nets/mobilenet/mobilenet_v3.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_common",
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "mobilenet_v2_test",
    srcs = ["nets/mobilenet/mobilenet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet",
        ":mobilenet_common",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/six",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "mobilenet_v3_test",
    srcs = ["nets/mobilenet/mobilenet_v3_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//testing/pybase:parameterized",
        "//third_party/py/absl/testing:absltest",
        # "//tensorflow",
    ],
)

py_library(
    name = "mobilenet",
    deps = [
        ":mobilenet_v1",
        ":mobilenet_v2",
        ":mobilenet_v3",
    ],
)

py_test(  # py2and3_test
    name = "mobilenet_v1_test",
    size = "large",
    srcs = ["nets/mobilenet_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_v1",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_binary(
    name = "mobilenet_v1_train",
    srcs = ["nets/mobilenet_v1_train.py"],
    python_version = "PY3",
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/contrib/quantize:quantize_graph",
    ],
)

py_binary(
    name = "mobilenet_v1_eval",
    srcs = ["nets/mobilenet_v1_eval.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/contrib/quantize:quantize_graph",
    ],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nets/nasnet/nasnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "nasnet",
    srcs = ["nets/nasnet/nasnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/contrib/training:training_py",
    ],
)

py_test(  # py2and3_test
    name = "nasnet_utils_test",
    size = "medium",
    srcs = ["nets/nasnet/nasnet_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet_utils",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
    ],
)

py_test(  # py2and3_test
    name = "nasnet_test",
    size = "large",
    srcs = ["nets/nasnet/nasnet_test.py"],
    shard_count = 10,
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "pnasnet",
    srcs = ["nets/nasnet/pnasnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet",
        ":nasnet_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/contrib/training:training_py",
    ],
)

py_test(  # py2and3_test
    name = "pnasnet_test",
    size = "large",
    srcs = ["nets/nasnet/pnasnet_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":pnasnet",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["nets/overfeat.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "overfeat_test",
    size = "medium",
    srcs = ["nets/overfeat_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":overfeat",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "pix2pix",
    srcs = ["nets/pix2pix.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "pix2pix_test",
    srcs = ["nets/pix2pix_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pix2pix",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "resnet_utils",
    srcs = ["nets/resnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "resnet_v1",
    srcs = ["nets/resnet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "resnet_v1_test",
    size = "medium",
    timeout = "long",
    srcs = ["nets/resnet_v1_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v1",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "resnet_v2",
    srcs = ["nets/resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "resnet_v2_test",
    size = "medium",
    srcs = ["nets/resnet_v2_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v2",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//numpy",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "s3dg",
    srcs = ["nets/s3dg.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":i3d_utils",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "s3dg_test",
    size = "large",
    srcs = ["nets/s3dg_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":s3dg",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/six",
        # "//tensorflow",
    ],
)

py_library(
    name = "vgg",
    srcs = ["nets/vgg.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "vgg_test",
    size = "medium",
    srcs = ["nets/vgg_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vgg",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
    ],
)

py_library(
    name = "nets_factory",
    srcs = ["nets/nets_factory.py"],
    deps = [
        ":nets",
        "//third_party/py/tf_slim:slim",
    ],
)

py_test(  # py2and3_test
    name = "nets_factory_test",
    size = "large",
    srcs = ["nets/nets_factory_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":nets_factory",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
    ],
)

pytype_strict_binary(
    name = "post_training_quantization",
    srcs = ["nets/post_training_quantization.py"],
    python_version = "PY3",
    deps = [
        ":nets_factory",
        ":preprocessing_factory",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        # "//tensorflow",
        # "//tensorflow_datasets",
    ],
)

py_library(
    name = "train_image_classifier_lib",
    srcs = ["train_image_classifier.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_factory",
        ":model_deploy",
        ":nets_factory",
        ":preprocessing_factory",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/contrib/quantize:quantize_graph",
    ],
)

py_binary(
    name = "train_image_classifier",
    srcs = ["train_image_classifier.py"],
    # WARNING: not supported in bazel; will be commented out by copybara.
    # paropts = ["--compress"],
    python_version = "PY3",
    deps = [
        ":train_image_classifier_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_library(
    name = "eval_image_classifier_lib",
    srcs = ["eval_image_classifier.py"],
    deps = [
        ":dataset_factory",
        ":nets_factory",
        ":preprocessing_factory",
        # "//tensorflow",
        "//third_party/py/tf_slim:slim",
        # "//tensorflow/contrib/quantize:quantize_graph",
    ],
)

py_binary(
    name = "eval_image_classifier",
    srcs = ["eval_image_classifier.py"],
    python_version = "PY3",
    deps = [
        ":eval_image_classifier_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_binary(
    name = "export_inference_graph",
    srcs = ["export_inference_graph.py"],
    # WARNING: not supported in bazel; will be commented out by copybara.
    # paropts = ["--compress"],
    python_version = "PY3",
    deps = [
        ":export_inference_graph_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_library(
    name = "export_inference_graph_lib",
    srcs = ["export_inference_graph.py"],
    deps = [
        ":dataset_factory",
        ":nets_factory",
        # "//tensorflow",
        # "//tensorflow/contrib/quantize:quantize_graph",
        # "//tensorflow/python:platform",
    ],
)

py_test(  # py2and3_test
    name = "export_inference_graph_test",
    size = "medium",
    srcs = ["export_inference_graph_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
    ],
    deps = [
        ":export_inference_graph_lib",
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        # "//tensorflow",
        # "//tensorflow/python:platform",
    ],
)
