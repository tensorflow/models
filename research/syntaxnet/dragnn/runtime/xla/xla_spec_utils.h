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

// Utils for working with specifications of XLA-based DRAGNN runtime models.

#ifndef DRAGNN_RUNTIME_XLA_XLA_SPEC_UTILS_H_
#define DRAGNN_RUNTIME_XLA_XLA_SPEC_UTILS_H_

#include <string>

#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// The name, file format, record format, and file suffix of the resource that
// contains the frozen TF GraphDef for each component.
extern const char *const kFrozenGraphDefResourceName;
extern const char *const kFrozenGraphDefResourceFileFormat;
extern const char *const kFrozenGraphDefResourceRecordFormat;
extern const char *const kFrozenGraphDefResourceFileSuffix;

// Returns the model name specified in |component_spec|, or the empty string
// if none is present.
string ModelNameForComponent(const ComponentSpec &component_spec);

// If |cell_subgraph_spec| is non-null, fills in |cell_subgraph_spec| from
// the |component_spec|. Returns non-OK when there is no CellSubgraphSpec
// present.
tensorflow::Status GetCellSubgraphSpecForComponent(
    const ComponentSpec &component_spec, CellSubgraphSpec *cell_subgraph_spec);

// Points |frozen_graph_def_resource| to the resource in the |component_spec|
// that specifies the frozen GraphDef.  On error, returns non-OK and modifies
// nothing.
tensorflow::Status LookupFrozenGraphDefResource(
    const ComponentSpec &component_spec,
    const Resource **frozen_graph_def_resource);

// Adds a resource to the |component_spec| that specifies the frozen GraphDef
// at the |path|.  On error, returns non-OK and modifies nothing.
tensorflow::Status AddFrozenGraphDefResource(const string &path,
                                             ComponentSpec *component_spec);

// Returns the name of the Xla input for the ID of the |index|'th feature in
// the |channel_id|'th fixed feature channel.
string MakeXlaInputFixedFeatureIdName(int channel_id, int index);

// Returns the names of the Xla inputs for the source activation vector and
// out-of-bounds indicator of the |channel_id|'th linked feature channel.
string MakeXlaInputLinkedActivationVectorName(int channel_id);
string MakeXlaInputLinkedOutOfBoundsIndicatorName(int channel_id);

// Returns the name of the Xla input for the hard-coded recurrent layer named
// |layer_name|.
string MakeXlaInputRecurrentLayerName(const string &layer_name);

// Returns the name of the Xla input for the generic layer named |layer_name|.
string MakeXlaInputLayerName(const string &layer_name);

// Returns the name of the Xla output for the layer named |layer_name|.
string MakeXlaOutputLayerName(const string &layer_name);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_XLA_SPEC_UTILS_H_
