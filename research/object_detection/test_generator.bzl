"""Macros to generate test targets for {TF1, TF2} x {TPU, CPU} combination."""

def generate_test_rules(
        name,
        srcs,
        deps,
        create_tpu_test = True,
        tf_version = None,
        **kwargs):
    """Generates test rule for {TF1, TF2} x {TPU, CPU} combinations.

    Usage:
    ------
    To generate test targets for {TF1, TF2} x {TPU, CPU} add this rule to BUILD
    file:

    generate_test_rules(
        name='my_test',
        srcs=[my_test.py],
        deps=[
            //dep1,
            ...
        ],
    )

    This produces 4 py test targets by default:
    * my_test.tf1
    * my_test.tpu.tf1
    * my_test.tf2
    * my_test.tpu.tf2

    The test source file must determine Tensorflow version and availibility of
    TPU to set up test appropriately. There is no need to add any dependency to
    disable TF2 or enable TPUs.

    Subclass third_party/tensorflow_models/object_detection/utils/test_case.py
    to set up tests easily.

    To list the generated test targets run:
    blaze query --output=build 'attr(generator_function, generate_test_rules,\
        //my/path:all)' | grep '^ *name' | cut -d'=' -f2

    To see the full definition of the generated test targets run:
    blaze query --output=build 'attr(generator_function, generate_test_rules,\
        //my/path:all)'

    Args:
        name: Name of the test.
        srcs: List of source files.
        deps: List of dependencies.
        create_tpu_test: Whether to create tpu tests. True by default.
        tf_version: Version of tf to test with. If unspecified, creates tests
            for both TF1 and TF2. Valid values for this argument are: None,
            "tf1_only", or "tf2_only".
        **kwargs: Additional py_test keyword arguments.
    """
    name_prefix = name
    if tf_version != "tf2_only":
        native.py_test(
            name = ".".join([name_prefix, "tf1"]),
            main = srcs[0],
            srcs = srcs,
            deps = deps + ["//learning/brain/public:disable_tf2"],
            python_version = "PY3",
            **kwargs
        )
        if create_tpu_test:
            native.py_test(
                name = ".".join([name_prefix, "tpu", "tf1"]),
                main = srcs[0],
                srcs = srcs,
                args = ["--register_deepsea_platform"],
                tags = ["requires-jellyfish"],
                deps = deps + [
                    "//learning/brain/public:disable_tf2",
                    "//learning/brain/google/xla:deepsea_hardware_device",
                ],
                python_version = "PY3",
                **kwargs
            )
    if tf_version != "tf1_only":
        native.py_test(
            name = ".".join([name_prefix, "tf2"]),
            main = srcs[0],
            srcs = srcs,
            deps = deps,
            python_version = "PY3",
            **kwargs
        )
        if create_tpu_test:
            native.py_test(
                name = ".".join([name_prefix, "tpu", "tf2"]),
                main = srcs[0],
                srcs = srcs,
                args = ["--register_deepsea_platform"],
                tags = ["requires-jellyfish"],
                deps = deps + ["//learning/brain/google/xla:deepsea_hardware_device"],
                python_version = "PY3",
                **kwargs
            )
