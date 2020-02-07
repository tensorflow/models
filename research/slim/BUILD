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
    python_version = "PY2",
    deps = [
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
    deps = [
        ":dataset_utils",
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
    deps = [
        ":dataset_utils",
        "//third_party/py/PIL:pil",
        "//third_party/py/contextlib2",
        # "//tensorflow",
    ],
)

py_library(
    name = "download_and_convert_visualwakewords",
    srcs = ["datasets/download_and_convert_visualwakewords.py"],
    deps = [
        ":download_and_convert_visualwakewords_lib",
        # "//tensorflow",
    ],
)

py_binary(
    name = "download_and_convert_data",
    srcs = ["download_and_convert_data.py"],
    python_version = "PY2",
    deps = [
        ":download_and_convert_cifar10",
        ":download_and_convert_flowers",
        ":download_and_convert_mnist",
        ":download_and_convert_visualwakewords",
        # "//tensorflow",
    ],
)

py_library(
    name = "cifar10",
    srcs = ["datasets/cifar10.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "flowers",
    srcs = ["datasets/flowers.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "imagenet",
    srcs = ["datasets/imagenet.py"],
    deps = [
        ":dataset_utils",
        "//third_party/py/six",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "mnist",
    srcs = ["datasets/mnist.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "visualwakewords",
    srcs = ["datasets/visualwakewords.py"],
    deps = [
        ":dataset_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "model_deploy_test",
    srcs = ["deployment/model_deploy_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":model_deploy",
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/layers:layers_py",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "cifarnet_preprocessing",
    srcs = ["preprocessing/cifarnet_preprocessing.py"],
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "vgg_preprocessing",
    srcs = ["preprocessing/vgg_preprocessing.py"],
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "alexnet_test",
    size = "medium",
    srcs = ["nets/alexnet_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":alexnet",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "cifarnet",
    srcs = ["nets/cifarnet.py"],
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "cyclegan",
    srcs = ["nets/cyclegan.py"],
    deps = [
        # "//numpy",
        "//third_party/py/six",
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/layers:layers_py",
        # "//tensorflow/contrib/util:util_py",
    ],
)

py_test(
    name = "cyclegan_test",
    srcs = ["nets/cyclegan_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":cyclegan",
        # "//tensorflow",
    ],
)

py_library(
    name = "dcgan",
    srcs = ["nets/dcgan.py"],
    deps = [
        "//third_party/py/six",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "dcgan_test",
    srcs = ["nets/dcgan_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":dcgan",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "i3d_test",
    size = "large",
    srcs = ["nets/i3d_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":i3d",
        # "//tensorflow",
    ],
)

py_library(
    name = "i3d_utils",
    srcs = ["nets/i3d_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/layers:layers_py",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "inception_v1",
    srcs = ["nets/inception_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "inception_v2",
    srcs = ["nets/inception_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "inception_v3",
    srcs = ["nets/inception_v3.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "inception_v4",
    srcs = ["nets/inception_v4.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "inception_resnet_v2",
    srcs = ["nets/inception_resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "inception_v1_test",
    size = "large",
    srcs = ["nets/inception_v1_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
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
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "inception_v3_test",
    size = "large",
    srcs = ["nets/inception_v3_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "inception_v4_test",
    size = "large",
    srcs = ["nets/inception_v4_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "inception_resnet_v2_test",
    size = "large",
    srcs = ["nets/inception_resnet_v2_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "lenet",
    srcs = ["nets/lenet.py"],
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "mobilenet_v1",
    srcs = ["nets/mobilenet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/layers:layers_py",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = ["nets/mobilenet/mobilenet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_common",
        # "//tensorflow",
        # "//tensorflow/contrib/layers:layers_py",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/slim",
    ],
)

py_test(  # py2and3_test
    name = "mobilenet_v2_test",
    srcs = ["nets/mobilenet/mobilenet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet",
        ":mobilenet_common",
        "//third_party/py/six",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(  # py2and3_test
    name = "mobilenet_v3_test",
    srcs = ["nets/mobilenet/mobilenet_v3_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet",
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

py_test(
    name = "mobilenet_v1_test",
    size = "large",
    srcs = ["nets/mobilenet_v1_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":mobilenet_v1",
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_binary(
    name = "mobilenet_v1_train",
    srcs = ["nets/mobilenet_v1_train.py"],
    python_version = "PY2",
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        # "//tensorflow",
        # "//tensorflow/contrib/quantize:quantize_graph",
        # "//tensorflow/contrib/slim",
    ],
)

py_binary(
    name = "mobilenet_v1_eval",
    srcs = ["nets/mobilenet_v1_eval.py"],
    python_version = "PY2",
    deps = [
        ":dataset_factory",
        ":mobilenet_v1",
        ":preprocessing_factory",
        # "//tensorflow",
        # "//tensorflow/contrib/quantize:quantize_graph",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nets/nasnet/nasnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "nasnet",
    srcs = ["nets/nasnet/nasnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/layers:layers_py",
        # "//tensorflow/contrib/slim",
        # "//tensorflow/contrib/training:training_py",
    ],
)

py_test(
    name = "nasnet_utils_test",
    size = "medium",
    srcs = ["nets/nasnet/nasnet_utils_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet_utils",
        # "//tensorflow",
    ],
)

py_test(
    name = "nasnet_test",
    size = "large",
    srcs = ["nets/nasnet/nasnet_test.py"],
    python_version = "PY2",
    shard_count = 10,
    srcs_version = "PY2AND3",
    deps = [
        ":nasnet",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/slim",
        # "//tensorflow/contrib/training:training_py",
    ],
)

py_test(
    name = "pnasnet_test",
    size = "large",
    srcs = ["nets/nasnet/pnasnet_test.py"],
    python_version = "PY2",
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":pnasnet",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["nets/overfeat.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(  # py2and3_test
    name = "overfeat_test",
    size = "medium",
    srcs = ["nets/overfeat_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":overfeat",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "pix2pix",
    srcs = ["nets/pix2pix.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/layers:layers_py",
    ],
)

py_test(  # py2and3_test
    name = "pix2pix_test",
    srcs = ["nets/pix2pix_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pix2pix",
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
    ],
)

py_library(
    name = "resnet_utils",
    srcs = ["nets/resnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "resnet_v1",
    srcs = ["nets/resnet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "resnet_v1_test",
    size = "medium",
    timeout = "long",
    srcs = ["nets/resnet_v1_test.py"],
    python_version = "PY2",
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v1",
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "resnet_v2",
    srcs = ["nets/resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(
    name = "resnet_v2_test",
    size = "medium",
    srcs = ["nets/resnet_v2_test.py"],
    python_version = "PY2",
    shard_count = 2,
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v2",
        # "//numpy",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "s3dg",
    srcs = ["nets/s3dg.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":i3d_utils",
        # "//tensorflow",
        # "//tensorflow/contrib/framework:framework_py",
        # "//tensorflow/contrib/layers:layers_py",
    ],
)

py_test(
    name = "s3dg_test",
    size = "large",
    srcs = ["nets/s3dg_test.py"],
    python_version = "PY2",
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":s3dg",
        # "//tensorflow",
    ],
)

py_library(
    name = "vgg",
    srcs = ["nets/vgg.py"],
    srcs_version = "PY2AND3",
    deps = [
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_test(  # py2and3_test
    name = "vgg_test",
    size = "medium",
    srcs = ["nets/vgg_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vgg",
        # "//tensorflow",
        # "//tensorflow/contrib/slim",
    ],
)

py_library(
    name = "nets_factory",
    srcs = ["nets/nets_factory.py"],
    deps = [
        ":nets",
        # "//tensorflow/contrib/slim",
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
        # "//tensorflow/contrib/quantize:quantize_graph",
        # "//tensorflow/contrib/slim",
    ],
)

py_binary(
    name = "train_image_classifier",
    srcs = ["train_image_classifier.py"],
    # WARNING: not supported in bazel; will be commented out by copybara.
    # paropts = ["--compress"],
    python_version = "PY2",
    deps = [
        ":train_image_classifier_lib",
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
        # "//tensorflow/contrib/quantize:quantize_graph",
        # "//tensorflow/contrib/slim",
    ],
)

py_binary(
    name = "eval_image_classifier",
    srcs = ["eval_image_classifier.py"],
    python_version = "PY2",
    deps = [
        ":eval_image_classifier_lib",
    ],
)

py_binary(
    name = "export_inference_graph",
    srcs = ["export_inference_graph.py"],
    # WARNING: not supported in bazel; will be commented out by copybara.
    # paropts = ["--compress"],
    python_version = "PY2",
    deps = [":export_inference_graph_lib"],
)

py_library(
    name = "export_inference_graph_lib",
    srcs = ["export_inference_graph.py"],
    deps = [
        ":dataset_factory",
        ":nets_factory",
        # "//tensorflow",
        # "//tensorflow/contrib/quantize:quantize_graph",
        # "//tensorflow/contrib/slim",
        # "//tensorflow/python:platform",
    ],
)

py_test(
    name = "export_inference_graph_test",
    size = "medium",
    srcs = ["export_inference_graph_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "manual",
    ],
    deps = [
        ":export_inference_graph_lib",
        # "//tensorflow",
        # "//tensorflow/python:platform",
    ],
)
