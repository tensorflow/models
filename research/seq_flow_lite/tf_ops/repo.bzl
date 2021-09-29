"""Reverb custom external dependencies."""

# Sanitize a dependency so that it works correctly from code that includes
# reverb as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def get_python_path(ctx):
    path = ctx.os.environ.get("PYTHON_BIN_PATH")
    if not path:
        fail(
            "Could not get environment variable PYTHON_BIN_PATH.  " +
            "Check your .bazelrc file.",
        )
    return path

def _find_tf_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_tf_lib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_lib())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_numpy_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import numpy; import sys; " +
            "sys.stdout.write(numpy.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate numpy includes path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_python_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "from distutils import sysconfig; import sys; " +
            "sys.stdout.write(sysconfig.get_python_inc())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate python includes path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_python_solib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import sys; vi = sys.version_info; " +
            "sys.stdout.write('python{}.{}'.format(vi.major, vi.minor))",
        ],
    )
    if exec_result.return_code != 0:
        fail("Could not locate python shared library path:\n{}"
            .format(exec_result.stderr))
    version = exec_result.stdout.splitlines()[-1]
    basename = "lib{}.so".format(version)
    exec_result = repo_ctx.execute(
        ["{}-config".format(version), "--configdir"],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate python shared library path:\n{}"
            .format(exec_result.stderr))
    solib_dir = exec_result.stdout.splitlines()[-1]
    full_path = repo_ctx.path("{}/{}".format(solib_dir, basename))
    if not full_path.exists:
        fail("Unable to find python shared library file:\n{}/{}"
            .format(solib_dir, basename))
    return struct(dir = solib_dir, basename = basename)

def _eigen_archive_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tf_includes/Eigen/**/*.h",
                 "tf_includes/Eigen/**",
                 "tf_includes/unsupported/Eigen/**/*.h",
                 "tf_includes/unsupported/Eigen/**"]),
    # https://groups.google.com/forum/#!topic/bazel-discuss/HyyuuqTxKok
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _nsync_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path + "/external", "nsync_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["nsync_includes/nsync/public/*.h"]),
    includes = ["nsync_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _zlib_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/zlib",
        "zlib",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["zlib/**/*.h"]),
    includes = ["zlib"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _snappy_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/snappy",
        "snappy",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["snappy/*.h"]),
    includes = ["snappy"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _protobuf_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.symlink(Label("//third_party:protobuf.BUILD"), "BUILD")

def _tensorflow_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tensorflow_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(
        [
            "tensorflow_includes/**/*.h",
            "tensorflow_includes/third_party/eigen3/**",
        ],
        exclude = ["tensorflow_includes/absl/**/*.h"],
    ),
    includes = ["tensorflow_includes"],
    deps = [
        "@eigen_archive//:eigen3",
        "@protobuf_archive//:includes",
        "@zlib_includes//:includes",
        "@snappy_includes//:includes",
    ],
    visibility = ["//visibility:public"],
)
filegroup(
    name = "protos",
    srcs = glob(["tensorflow_includes/**/*.proto"]),
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _tensorflow_solib_repo_impl(repo_ctx):
    tf_lib_path = _find_tf_lib_path(repo_ctx)
    repo_ctx.symlink(tf_lib_path, "tensorflow_solib")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "framework_lib",
    srcs = ["tensorflow_solib/libtensorflow_framework.so.2"],
    deps = ["@python_includes", "@python_includes//:numpy_includes"],
    visibility = ["//visibility:public"],
)
""",
    )

def _python_includes_repo_impl(repo_ctx):
    python_include_path = _find_python_include_path(repo_ctx)
    python_solib = _find_python_solib_path(repo_ctx)
    repo_ctx.symlink(python_include_path, "python_includes")
    numpy_include_path = _find_numpy_include_path(repo_ctx)
    repo_ctx.symlink(numpy_include_path, "numpy_includes")
    repo_ctx.symlink(
        "{}/{}".format(python_solib.dir, python_solib.basename),
        python_solib.basename,
    )

    # Note, "@python_includes" is a misnomer since we include the
    # libpythonX.Y.so in the srcs, so we can get access to python's various
    # symbols at link time.
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "python_includes",
    hdrs = glob(["python_includes/**/*.h"]),
    srcs = ["{}"],
    includes = ["python_includes"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "numpy_includes",
    hdrs = glob(["numpy_includes/**/*.h"]),
    includes = ["numpy_includes"],
    visibility = ["//visibility:public"],
)
""".format(python_solib.basename),
        executable = False,
    )

def cc_tf_configure():
    """Autoconf pre-installed tensorflow repo."""
    make_nsync_repo = repository_rule(
        implementation = _nsync_includes_repo_impl,
    )
    make_nsync_repo(name = "nsync_includes")
    make_zlib_repo = repository_rule(
        implementation = _zlib_includes_repo_impl,
    )
    make_zlib_repo(name = "zlib_includes")
    make_snappy_repo = repository_rule(
        implementation = _snappy_includes_repo_impl,
    )
    make_snappy_repo(name = "snappy_includes")
    make_protobuf_repo = repository_rule(
        implementation = _protobuf_includes_repo_impl,
    )
    make_protobuf_repo(name = "protobuf_archive")
    make_tfinc_repo = repository_rule(
        implementation = _tensorflow_includes_repo_impl,
    )
    make_tfinc_repo(name = "tensorflow_includes")
    make_tflib_repo = repository_rule(
        implementation = _tensorflow_solib_repo_impl,
    )
    make_tflib_repo(name = "tensorflow_solib")
    make_python_inc_repo = repository_rule(
        implementation = _python_includes_repo_impl,
    )
    make_python_inc_repo(name = "python_includes")

def _reverb_protoc_archive(ctx):
    version = ctx.attr.version
    sha256 = ctx.attr.sha256

    override_version = ctx.os.environ.get("REVERB_PROTOC_VERSION")
    if override_version:
        sha256 = ""
        version = override_version

    urls = [
        "https://github.com/protocolbuffers/protobuf/releases/download/v%s/protoc-%s-linux-x86_64.zip" % (version, version),
    ]
    ctx.download_and_extract(
        url = urls,
        sha256 = sha256,
    )

    ctx.file(
        "BUILD",
        content = """
filegroup(
    name = "protoc_bin",
    srcs = ["bin/protoc"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

reverb_protoc_archive = repository_rule(
    implementation = _reverb_protoc_archive,
    attrs = {
        "version": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
    },
)

def reverb_protoc_deps(version, sha256):
    reverb_protoc_archive(name = "protobuf_protoc", version = version, sha256 = sha256)
