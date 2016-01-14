py_library(
    name = "data_utils",
    srcs = [
        "data_utils.py",
    ],
    deps = [
        "//file/colossus/public:cns",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "neural_gpu",
    srcs = [
        "neural_gpu.py",
    ],
    deps = [
        ":data_utils",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_binary(
    name = "neural_gpu_trainer",
    srcs = [
        "neural_gpu_trainer.py",
    ],
    launcher = "//devtools/python/launcher",
    malloc = "//tcmalloc:tcmalloc_or_debug",
    deps = [
        ":neural_gpu",
        "//file/colossus/public:cns",
        "//net/proto2/python/public:use_fast_cpp_protos",
        "//third_party/py/Tkinter",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)
