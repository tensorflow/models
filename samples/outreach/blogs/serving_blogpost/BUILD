package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "train_export_temp_model",
    srcs = [
        "train_export_temp_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "temperature_client",
    srcs = [
        "temperature_client.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
