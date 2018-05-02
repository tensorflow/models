// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <stddef.h>
#include <memory>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/xla/sequence_xla_dynamic_component_mixin.h"
#include "dragnn/runtime/xla/xla_dynamic_component_base.h"
#include "dragnn/runtime/xla/xla_graph_utils.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// An XLA-based version of DynamicComponent using the XLA JIT API.

//
// It uses the XLA JIT API to compile the graph, and uses the frozen GraphDef
// referred to in the component spec.
class XlaDynamicComponent : public XlaDynamicComponentBase {
 protected:
  // Unlike other specializations, this component will only be active if the
  // spec is explicitly modified to support XLA (and frozen graph resources are
  // generated).
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "XlaDynamicComponent";
  }
  bool PreferredTo(const Component &other) const override { return false; }

  // Gets the frozen GraphDef using the |component_spec| and compiles it.
  // The |cell_subgraph_spec| contained within it is filled in. On error,
  // returns non-OK.
  tensorflow::Status InitializeFromComponentSpec(
      const ComponentSpec &component_spec,
      CellSubgraphSpec *cell_subgraph_spec) override;

  const tensorflow::XlaCompiledCpuFunction::StaticData &XlaStaticData()
      const override {
    if (jit_ == nullptr) {
      LOG(FATAL) << "XlaStaticData() called before "
                    "InitializeFromComponentSpec() for component "
                 << name();
    }
    return jit_->StaticData();
  }

 private:
  // Cell that contains the compiled code for this component.
  std::unique_ptr<tensorflow::XlaJitCompiledCpuFunction> jit_;
};

tensorflow::Status XlaDynamicComponent::InitializeFromComponentSpec(
    const ComponentSpec &component_spec, CellSubgraphSpec *cell_subgraph_spec) {
  const Resource *resource = nullptr;
  TF_RETURN_IF_ERROR(LookupFrozenGraphDefResource(component_spec, &resource));
  const string &frozen_graph_def_path = resource->part(0).file_pattern();
  tensorflow::GraphDef frozen_graph_def;
  TF_RETURN_IF_ERROR(
      LoadFrozenGraphDef(frozen_graph_def_path, &frozen_graph_def));

  // Gets the CellSubgraphSpec from the frozen GraphDef and constructs
  // the XLA Config required for compilation.
  tensorflow::tf2xla::Config xla_config;
  TF_RETURN_IF_ERROR(GetSpecAndMakeXlaConfig(frozen_graph_def,
                                             cell_subgraph_spec, &xla_config));

  // Compiles the cell.
  TF_ASSIGN_OR_RETURN(
      jit_, tensorflow::XlaJitCompiledCpuFunction::Compile(
                frozen_graph_def, xla_config, xla::ExecutableBuildOptions()));

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(XlaDynamicComponent);

// Sequence-based version of the above.
using SequenceXlaDynamicComponent =
    SequenceXlaDynamicComponentMixin<XlaDynamicComponent>;

DRAGNN_RUNTIME_REGISTER_COMPONENT(SequenceXlaDynamicComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
