# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")
load("@protobuf_archive//:protobuf.bzl", "py_proto_library")


def if_cuda(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with CUDA."""
    return select({
        "@local_config_cuda//cuda:using_nvcc": if_true,
        "@local_config_cuda//cuda:using_clang": if_true,
        "//conditions:default": if_false
    })

def tf_copts():
  return (["-fno-exceptions", "-DEIGEN_AVOID_STL_ARRAY",] +
          if_cuda(["-DGOOGLE_CUDA=1"]) +
          select({"@org_tensorflow//tensorflow:darwin": [],
                  "//conditions:default": ["-pthread"]}))

def tf_proto_library(name, srcs=[], has_services=False,
                     deps=[], visibility=None, testonly=0,
                     cc_api_version=2, go_api_version=2,
                     java_api_version=2,
                     py_api_version=2):
  native.filegroup(name=name + "_proto_srcs",
                   srcs=srcs,
                   testonly=testonly,)

  cc_proto_library(name=name,
                   srcs=srcs,
                   deps=deps,
                   cc_libs = ["@protobuf_archive//:protobuf"],
                   protoc="@protobuf_archive//:protoc",
                   default_runtime="@protobuf_archive//:protobuf",
                   testonly=testonly,
                   visibility=visibility,)

def tf_proto_library_py(name, srcs=[], deps=[], visibility=None, testonly=0):
  py_proto_library(name=name,
                   srcs=srcs,
                   srcs_version = "PY2AND3",
                   deps=deps,
                   default_runtime="@protobuf_archive//:protobuf_python",
                   protoc="@protobuf_archive//:protoc",
                   visibility=visibility,
                   testonly=testonly,)

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(op_lib_names):
  # Make library out of each op so it can also be used to generate wrappers
  # for various languages.
  for n in op_lib_names:
    native.cc_library(name=n + "_op_lib",
                      copts=tf_copts(),
                      srcs=["ops/" + n + ".cc"],
                      deps=(["@org_tensorflow//tensorflow/core:framework"]),
                      visibility=["//visibility:public"],
                      alwayslink=1,
                      linkstatic=1,)

# Invoke this rule in .../tensorflow/python to build the wrapper library.
def tf_gen_op_wrapper_py(name, out=None, hidden=[], visibility=None, deps=[],
                         require_shape_functions=False):
  # Construct a cc_binary containing the specified ops.
  tool_name = "gen_" + name + "_py_wrappers_cc"
  if not deps:
    deps = ["//tensorflow/core:" + name + "_op_lib"]
  native.cc_binary(
      name = tool_name,
      linkopts = ["-lm"],
      copts = tf_copts(),
      linkstatic = 1,   # Faster to link this one-time-use binary dynamically
      deps = (["@org_tensorflow//tensorflow/core:framework",
               "@org_tensorflow//tensorflow/python:python_op_gen_main"] + deps),
  )

  # Invoke the previous cc_binary to generate a python file.
  if not out:
    out = "ops/gen_" + name + ".py"

  native.genrule(
      name=name + "_pygenrule",
      outs=[out],
      tools=[tool_name],
      cmd=("$(location " + tool_name + ") " + ",".join(hidden)
           + " " + ("1" if require_shape_functions else "0") + " > $@"))

  # Make a py_library out of the generated python file.
  native.py_library(name=name,
                    srcs=[out],
                    srcs_version="PY2AND3",
                    visibility=visibility,
                    deps=[
                        "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers",
                    ],)
