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
"""Build rules for Syntaxnet."""

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    orig_tf_proto_library_cc = "tf_proto_library_cc",
)
load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    orig_tf_proto_library_py = "tf_proto_library_py",
)

# For some reason, tf_proto_library_cc() isn't obeying the default_visibility
# directive at the top of the build file. So just set it to public (which it is
# anyway).
def tf_proto_library_cc(name, visibility=[], **kwargs):
  visibility = visibility if visibility else ["//visibility:public"]
  return orig_tf_proto_library_cc(name, visibility=visibility, **kwargs)

def tf_proto_library_py(name, visibility=[], **kwargs):
  visibility = visibility if visibility else ["//visibility:public"]
  return orig_tf_proto_library_py(name, visibility=visibility, **kwargs)
