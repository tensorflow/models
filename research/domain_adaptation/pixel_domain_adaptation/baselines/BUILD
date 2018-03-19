licenses(["notice"])  # Apache 2.0

py_binary(
    name = "baseline_train",
    srcs = ["baseline_train.py"],
    deps = [
        
        "//domain_adaptation/datasets:dataset_factory",
        "//domain_adaptation/pixel_domain_adaptation:pixelda_model",
        "//domain_adaptation/pixel_domain_adaptation:pixelda_preprocess",
    ],
)

py_binary(
    name = "baseline_eval",
    srcs = ["baseline_eval.py"],
    deps = [
        
        "//domain_adaptation/datasets:dataset_factory",
        "//domain_adaptation/pixel_domain_adaptation:pixelda_model",
        "//domain_adaptation/pixel_domain_adaptation:pixelda_preprocess",
    ],
)
