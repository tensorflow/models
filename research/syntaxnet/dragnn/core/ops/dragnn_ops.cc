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

#include "tensorflow/core/framework/op.h"

namespace syntaxnet {
namespace dragnn {

REGISTER_OP("SetAssetDirectory")
    .Input("asset_directory: string")
    .Output("asset_directory_out: string")
    .SetIsStateful()
    .Doc(R"doc(
Override the paths to assets specified in the MasterSpec with the given
asset_directory. This op must be called before any calls to GetSession, as it
will create a new session pool with the overridden master spec.

asset_directory: The directory containing all the assets. Note that all assets
    must be in a single flat directory.
asset_directory_out: The input, just as an output.
)doc");

REGISTER_OP("GetSession")
    .Input("container: string")
    .Attr("master_spec: string")
    .Attr("grid_point: string")
    .Output("handle: string")
    .SetIsStateful()
    .Doc(R"doc(
Given MasterSpec and GridPoint protos, outputs a handle to a ComputeSession.

container: A unique identifier for the ComputeSessionPool from which a
    ComputeSession will be allocated.
master_spec: A serialized syntaxnet.dragnn.MasterSpec proto.
grid_point: A serialized syntaxnet.dragnn.GridPoint proto.
handle: A string handle to a ComputeSession.
)doc");

REGISTER_OP("ReleaseSession").Input("handle: string").SetIsStateful().Doc(R"doc(
Given a ComputeSession, return it to the ComputeSession pool.

This ComputeSession will no longer be available after this op returns.

handle: A handle to a ComputeSession that will be returned to the backing pool.
)doc");

REGISTER_OP("GetSessionCounts")
    .Input("container: string")
    .Output("stats: int64")
    .SetIsStateful()
    .Doc(R"doc(
Given a container string, output session counts for that ComputeSessionPool.

container: A unique identifier for the ComputeSessionPool to analyze.
stats: A vector of stats. [0] is the total number of created sessions. [1] is
the number of sessions that are currently not in the pool.
)doc");

REGISTER_OP("InitComponentData")
    .Input("handle: string")
    .Input("beam_size: int32")
    .Attr("component: string")
    .Output("output_handle: string")
    .Doc(R"doc(
Initialize a component with the given beam size for a given ComputeSession.

handle: A handle to a ComputeSession.
beam_size: The size of the beam to use on the component.
component: The name of a Component instance, matching the ComponentSpec.name.
output_handle: The handle to the same ComputeSession after initialization.
)doc");

REGISTER_OP("BatchSize")
    .Input("handle: string")
    .Attr("component: string")
    .Output("batch_size: int32")
    .Doc(R"doc(
Given a ComputeSession and a component name,return the component batch size.

handle: A handle to a ComputeSession.
component: The name of a Component instance, matching the ComponentSpec.name.
batch_size: The size of the given component's batch.
)doc");

REGISTER_OP("SetTracing")
    .Input("handle: string")
    .Input("tracing_on: bool")
    .Attr("component: string = 'NOT_USED_FOR_THIS_OP'")
    .Output("output_handle: string")
    .Doc(R"doc(
Given a ComputeSession, turns on or off tracing for all components.

handle: A handle to a ComputeSession.
tracing_on: Whether or not to record traces.
output_handle: The handle to the same ComputeSession, with the tracing status changed.
)doc");

REGISTER_OP("AttachDataReader")
    .Input("handle: string")
    .Input("input_spec: string")
    .Attr("component: string = 'NOT_USED_FOR_THIS_OP'")
    .Output("output_handle: string")
    .Doc(R"doc(
Given a ComputeSession, attach a data source.

This op is agnostic to the type of input data. The vector of input strings is
interpreted by the backend.

handle: A handle to a ComputeSession.
input_spec: A vector of strings, where each string represents one batch item.
output_handle: The handle to the same ComputeSession after attachment.
)doc");

REGISTER_OP("AdvanceFromOracle")
    .Input("handle: string")
    .Attr("component: string")
    .Output("output_handle: string")
    .Doc(R"doc(
Given a ComputeSession and a Component name, advance the component via oracle.

handle: A handle to a ComputeSession.
component: The name of a Component instance, matching the ComponentSpec.name.
output_handle: The handle to the same ComputeSession after advancement.
)doc");

REGISTER_OP("AdvanceFromPrediction")
    .Input("handle: string")
    .Input("scores: float")
    .Attr("component: string")
    .Output("output_handle: string")
    .Doc(R"doc(
Given a ComputeSession, a Component name, and a score tensor, advance the state.

handle: A handle to a ComputeSession.
scores: A tensor of scores, ordered by {batch_size, beam_size, num_actions}.
component: The name of a Component instance, matching the ComponentSpec.name.
output_handle: A handle to the same ComputeSession after advancement.
)doc");

REGISTER_OP("ExtractFixedFeatures")
    .Input("handle: string")
    .Output("indices: int32")
    .Output("ids: int64")
    .Output("weights: float")
    .Attr("component: string")
    .Attr("channel_id: int")
    .Doc(R"doc(
Given a ComputeSession, Component, and channel index, output fixed features.

Fixed features returned as 3 vectors, 'indices', 'ids', and 'weights' of equal
length. 'ids' specifies which rows should be looked up in the embedding
matrix. 'weights' specifies a scale for each embedding vector. 'indices' is a
sorted vector that assigns the same index to embedding vectors that should be
summed together.

handle: A handle to a ComputeSession.
indices: The row to add the feature to.
ids: The indices into embedding matrices for each feature.
weights: The weight for each looked up feature.
component: The name of a Component instance, matching the ComponentSpec.name.
channel_id: The feature channel to extract features for.
)doc");

REGISTER_OP("ExtractLinkFeatures")
    .Input("handle: string")
    .Output("step_idx: int32")
    .Output("idx: int32")
    .Attr("component: string")
    .Attr("channel_id: int")
    .Doc(R"doc(
Given a ComputeSession, Component, and a channel index, outputs link features.

Output indices have shape {batch_size * beam_size * channel_size}.

handle: A handle to a ComputeSession.
step_idx: The step indices to read activations from.
idx: indices The index within a step to read the activations from.
component: The name of a Component instance, matching the ComponentSpec.name.
channel_id: The feature channel to extract features for.
)doc");

REGISTER_OP("EmitOracleLabels")
    .Input("handle: string")
    .Output("gold_labels: int32")
    .Attr("component: string")
    .Doc(R"doc(
Given a ComputeSession and Component, emit a vector of gold labels.

handle: A handle to a ComputeSession.
gold_labels: A [batch_size * beam_size] vector of gold labels for the current
             ComputeSession.
component: The name of a Component instance, matching the ComponentSpec.name.
)doc");

REGISTER_OP("EmitAllFinal")
    .Input("handle: string")
    .Output("all_final: bool")
    .Attr("component: string")
    .Doc(R"doc(
Given a ComputeSession and Component, returns whether the Component is final.

A component is considered final when all elements in the batch have beams
containing all final states.

handle: A handle to a ComputeSession.
all_final: Whether every element in the specified component is 'final'.
component: The name of a Component instance, matching the ComponentSpec.name.
)doc");

REGISTER_OP("WriteAnnotations")
    .Input("handle: string")
    .Output("output_handle: string")
    .Attr("component: string")
    .Doc(R"doc(
Given a ComputeSession, has the given component write out its annotations.

The annotations are written to the underlying data objects passed in at the
beginning of the computation.

handle: A handle to a ComputeSession.
output_handle: A handle to the same ComputeSession after writing.
component: The name of a Component instance, matching the ComponentSpec.name.
)doc");

REGISTER_OP("EmitAnnotations")
    .Input("handle: string")
    .Output("annotations: string")
    .Attr("component: string")
    .Doc(R"doc(
Given a ComputeSession, emits strings with final predictions for the model.

Predictions are given for each element in the final component's batch.

handle: A handle to a ComputeSession.
annotations: A vector of strings representing the annotated data.
component: The name of a Component instance, matching the ComponentSpec.name.
)doc");

REGISTER_OP("GetComponentTrace")
    .Input("handle: string")
    .Output("trace: string")
    .Attr("component: string")
    .Doc(R"doc(
Gets the raw MasterTrace proto for each batch, state, and beam slot.

handle: A handle to a ComputeSession.
trace: A vector of MasterTrace protos.
component: The name of a Component instance, matching the ComponentSpec.name.
)doc");

}  // namespace dragnn
}  // namespace syntaxnet
