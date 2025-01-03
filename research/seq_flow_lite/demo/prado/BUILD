# A demo app for invoking a PRADO TFLite model.

licenses(["notice"])

package(default_visibility = ["//:friends"])  # sequence projection

cc_binary(
    name = "prado_tflite_example",
    srcs = ["prado_tflite_example.cc"],
    data = [
        "data/tflite.fb",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "//tflite_ops:expected_value",  # sequence projection
        "//tflite_ops:quantization_util",  # sequence projection
        "//tflite_ops:sequence_string_projection",  # sequence projection
    ],
)
