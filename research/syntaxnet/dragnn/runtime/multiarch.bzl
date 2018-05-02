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

"""Build extension rules for handling multiple target architectures."""

# Build configs for specific CPU architectures. Each entry specified
# additional copts and tags.
# TODO(googleuser): Figure out a workaround for the shift
# instructions, and look for any other unsupported instructions.
MULTIARCH_CONFIGS = {
    "generic": {
        "copts": [],
        "tags": [],
    },
    "avx": {
        "copts": [
            "-msse4.2",
        ],
        "tags": [],
    },
    "avx2fma": {
        "copts": [
            "-msse4.2",
            "-mavx",
            "-mavx2",
            "-mfma",
        ],
        "tags": [
            "local",
            "manual",
        ],
    },
}

# List of targets which are built for multiple architectures. These
# dependencies in dragnn_cc_* build rules are replaced with one with the
# appropriate suffix, e.g. _multiarch_generic
MULTIARCH_TARGETS = [
    "//dragnn/runtime:biaffine_digraph_component",
    "//dragnn/runtime:bulk_dynamic_component",
    "//dragnn/runtime:bulk_feed_forward_network",
    "//dragnn/runtime:bulk_lstm_network",
    "//dragnn/runtime:feed_forward_network",
    "//dragnn/runtime:feed_forward_network_kernel",
    "//dragnn/runtime:feed_forward_network_layer",
    "//dragnn/runtime:fixed_embeddings",
    "//dragnn/runtime:linked_embeddings",
    "//dragnn/runtime:lstm_network",
    "//dragnn/runtime:lstm_network_kernel",
    "//dragnn/runtime:network_unit_base",
    "//dragnn/runtime:sequence_bulk_dynamic_component",
    "//dragnn/runtime:sequence_features",
    "//dragnn/runtime:sequence_links",
    "//dragnn/runtime:sequence_model",
    "//dragnn/runtime/lstm_cell:cell_function",
    "//dragnn/runtime/lstm_cell:test_helpers",
    "//dragnn/runtime/myelin:myelin_dynamic_component",
    "//dragnn/runtime/myelin:myelin_dynamic_component_base",
    "//dragnn/runtime/myelin:sequence_myelin_dynamic_component",
    "//dragnn/runtime/xla:sequence_xla_dynamic_component_mixin",
    "//dragnn/runtime/xla:testdata_simple_component_library",
    "//dragnn/runtime/xla:xla_aot_dynamic_component",
    "//dragnn/runtime/xla:xla_dynamic_component",
    "//dragnn/runtime/xla:xla_dynamic_component_base",
]

def multiarch_name(target_name, arch_name):
  """Generates the multiarch version of |target_name| given |arch_name|."""
  return target_name + '_multiarch_' + arch_name

def _is_multiarch(target):
  """Returns true if |target| is designated as a multiarch target."""
  return (target in MULTIARCH_TARGETS or
          ('//' + native.package_name() + target) in MULTIARCH_TARGETS)

def _dragnn_cc_multiarch_target(native_rule = None,
                                name = '',
                                target_arch = None,
                                target_suffix = '',
                                copts = [],
                                deps = [],
                                tags = [],
                                opts_self = False,
                                deps_transformer = None,
                                **kwargs):
  """Generates a target for multiple architectures.

  Using the |native_rule| (e.g. cc_library) to create a set of targets for
  all CPU architectures listed in MULTIARCH_CONFIGS, with added suffixes
  that designate the architecture.

  When |target_arch| is set, then only that single target is generated,
  and the name of the target is unchanged (no suffix is added).

  When |opts_self| is true, then the 'copts' entry in MULTIARCH_CONFIGS
  is additionally used to build this target.

  The 'tags' entry in MULTIARCH_CONFIGS are included in the build tags.

  Args:
    native_rule: The build rule used for all generated targets
    name: The original name of the build rule (without any suffix).
    target_arch: When set, only this architecture is targeted.
    target_suffix: Additional suffix to add after the architecture.
    copts: The original compilation options for this target.
    deps: The original dependencies for this target.
    tags: The original build tags for this target.
    opts_self: When true, additional copts are included.
    deps_transformer: When set, a function to apply to the multiarch deps.
    **kwargs: Additional args passed along to the build rule.
  """
  # Determine set of target architectures based on |target_arch|.
  if target_arch:
    if target_arch in MULTIARCH_CONFIGS:
      arch_items = [(target_arch, MULTIARCH_CONFIGS[target_arch])]
    else:
      fail('Unknown target_arch value: ' + target_arch)
  else:
    arch_items = MULTIARCH_CONFIGS.items()

  # There is one target for each architecture in |arch_items|.
  for arch, arch_config in arch_items:
    # Transform the multi-arch deps.
    multiarch_deps = [multiarch_name(dep, arch) if _is_multiarch(dep) else dep
                      for dep in deps]
    if deps_transformer:
      multiarch_deps = deps_transformer(multiarch_deps)

    native_rule(
        name = (name if target_arch else multiarch_name(name, arch)) + target_suffix,
        copts = copts + arch_config['copts'] if opts_self else copts,
        deps = multiarch_deps,
        tags = tags + arch_config['tags'],
        **kwargs)

def _dragnn_cc_multiarch_test_target(name = None,
                                     target_arch = None,
                                     **kwargs):
  """Test target wrapper which puts arch name before '_test'."""
  test_suffix = '_test'
  has_test_suffix = name.endswith(test_suffix)

  # Keeps _test at the end of the target name.
  test_name = name[:-len(test_suffix)] if has_test_suffix else name
  target_suffix = test_suffix if has_test_suffix else ''

  _dragnn_cc_multiarch_target(native_rule = native.cc_test,
                              name = test_name,
                              target_arch = target_arch,
                              target_suffix = target_suffix,
                              **kwargs)

  # When |target_arch| is set, the resulting test is named |name|.  Otherwise,
  # tests with arch-specific names are generated, and for convenience we add a
  # test_suite named |name| that runs the generic version of the test.
  if not target_arch:
    native.test_suite(
        name = name,
        tests = [multiarch_name(test_name, 'generic') + target_suffix])

def dragnn_cc_multiarch_library(**kwargs):
  """Similar to cc_library, but creates multiple architecture targets."""

  _dragnn_cc_multiarch_target(native_rule = native.cc_library,
                              **kwargs)

def dragnn_cc_multiarch_test(**kwargs):
  """Similar to cc_test, but creates multiple architecture targets."""
  _dragnn_cc_multiarch_test_target(**kwargs)

def dragnn_cc_multiarch_binary(**kwargs):
  """Similar to cc_binary, but creates multiple architecture targets."""
  _dragnn_cc_multiarch_target(native_rule = native.cc_binary,
                              **kwargs)

def dragnn_cc_library(target_arch = 'generic', **kwargs):
  """Similar to cc_library, but targets one specific architecture."""
  _dragnn_cc_multiarch_target(native_rule = native.cc_library,
                              target_arch = target_arch,
                              **kwargs)

def dragnn_cc_test(target_arch = 'generic', **kwargs):
  """Similar to cc_test, but targets one specific architecture."""
  _dragnn_cc_multiarch_test_target(target_arch = target_arch,
                                   **kwargs)

def dragnn_cc_binary(target_arch = 'generic', **kwargs):
  """Similar to cc_binary, but targets one specific architecture."""
  _dragnn_cc_multiarch_target(native_rule = native.cc_binary,
                              target_arch = target_arch,
                              **kwargs)
