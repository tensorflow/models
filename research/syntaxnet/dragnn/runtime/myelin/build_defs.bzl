# Copyright 2017 Google Inc. All Rights Reserved.
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
# =============================================================================

"""Build rules that restrict Myelin to supported environments.

As of this writing, SLING requires Linux and x86-64:
https://github.com/google/sling/blob/master/README.md#building

The technique used here is to replace the hdrs, srcs, and deps with appropriate
empty content when building in an unsupported environment.
"""

load(
    "//dragnn/runtime:multiarch.bzl",
    "dragnn_cc_multiarch_library",
    "dragnn_cc_multiarch_test",
)

def _if_supported(consequent, alternative=[]):
  """Returns the |consequent| iff the build environment supports Myelin."""
  return select({
      "@org_tensorflow//tensorflow:linux_x86_64": consequent,
      "//conditions:default": alternative,
  })

def _if_supported_test_deps(deps):
  """Like _if_supported, but returns appropriate fallback deps for a test."""
  return _if_supported(deps, ["//syntaxnet:test_main"])

def dragnn_myelin_cc_library(hdrs=[], srcs=[], deps=[], **kwargs):
  """Like cc_library, but reduces to a NOP in unsupported environments."""
  native.cc_library(
      hdrs = _if_supported(hdrs),
      srcs = _if_supported(srcs),
      deps = _if_supported(deps),
      **kwargs)

def dragnn_myelin_cc_test(srcs=[], deps=[], **kwargs):
  """Like cc_test, but reduces to a NOP in unsupported environments."""
  native.cc_test(
      srcs = _if_supported(srcs),
      deps = _if_supported_test_deps(deps),
      **kwargs)

# Implementation note: Bazel select()s are not resolved at the time that build
# rules are evaluated.  If we pass _if_supported(deps) into the multi-arch build
# rules (like we do for the native rules above), then the multi-arch rules break
# when they attempt to iterate over the deps---at that point, the deps are an
# unresolved select() that can't be iterated.  To get around this, we delay the
# select() by passing _if_supported into the multi-arch rule, which will apply
# it just before passing the deps to the native rule.

def dragnn_myelin_cc_multiarch_library(hdrs=[], srcs=[], **kwargs):
  """Multi-arch version of dragnn_myelin_cc_library."""
  dragnn_cc_multiarch_library(
      hdrs = _if_supported(hdrs),
      srcs = _if_supported(srcs),
      deps_transformer = _if_supported,
      **kwargs)

def dragnn_myelin_cc_multiarch_test(srcs=[], **kwargs):
  """Multi-arch version of dragnn_myelin_cc_test."""
  dragnn_cc_multiarch_test(
      srcs = _if_supported(srcs, []),
      deps_transformer = _if_supported_test_deps,
      **kwargs)
