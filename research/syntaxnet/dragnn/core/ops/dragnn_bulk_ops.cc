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

#include "dragnn/core/ops/shape_helpers.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace syntaxnet {
namespace dragnn {

REGISTER_OP("BulkFixedFeatures")
    .Input("handle: string")
    .Output("output_handle: string")
    .Output("indices: num_channels * int32")
    .Output("ids: num_channels * int64")
    .Output("weights: num_channels * float")
    .Output("num_steps: int32")
    .Attr("component: string")
    .Attr("num_channels: int")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int num_channels;
      TF_RETURN_IF_ERROR(context->GetAttr("num_channels", &num_channels));
      for (int i = 1; i <= 3 * num_channels; ++i) {
        VectorOutputShape(i, context);
      }
      ScalarOutputShape(3 * num_channels + 1, context);
      return ComputeSessionHandleInputAndOutputShape(context);
    })
    .Doc(R"doc(
Given a ComputeSession and a component, outputs fixed features for all steps.

This op outputs features for the entire oracle path of the component. Unlike
ExtractFixedFeatures, this op mutates the master state, advancing all of its
states until they are final. For every channel, indices[channel], ids[channel],
and weights[channel] have the same length, ie. the number of predicates,
ordered by batch, beam, step.

handle: A handle to a ComputeSession.
output_handle: A handle to the same ComputeSession after advancement.
indices: (num_channels vectors of int32) If indices[i] = j, then
  embedding_sum[j] += embedding_matrix[ids[i]] * weights[i].
ids: (num_channels vectors of int64) Ids to lookup in embedding matrices.
weights: (num_channels vectors of float) Weight for each embedding.
num_steps: (int32 scalar) The batch was unrolled for this many steps.
component: The name of a Component instance, matching the ComponentSpec.name.
num_channels: The number of FixedFeature channels.
)doc");

REGISTER_OP("BulkFixedEmbeddings")
    .Input("handle: string")
    .Input("embedding_matrix: num_channels * T")
    .Output("output_handle: string")
    .Output("embedding_vectors: T")
    .Output("num_steps: int32")
    .Attr("component: string")
    .Attr("num_channels: int")
    .Attr("T: type")
    .Attr("pad_to_batch: int=-1")
    .Attr("pad_to_steps: int=-1")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int num_channels;
      TF_RETURN_IF_ERROR(context->GetAttr("num_channels", &num_channels));
      for (int i = 1; i <= num_channels; ++i) {
        TF_RETURN_IF_ERROR(MatrixInputShape(i, context));
      }
      MatrixOutputShape(1, context);
      ScalarOutputShape(2, context);
      return ComputeSessionHandleInputAndOutputShape(context);
    })
    .Doc(R"doc(
This op is a more efficient version of BulkFixedFeatures.

It is intended to be run with large batch sizes at inference time. The op takes
a handle to ComputeSession and embedding matrices as tensor inputs, and directly
outputs concatenated embedding vectors.

handle: A handle to ComputeSession.
embedding_matrix: Embedding matrices.
output_handle: A handle to the same ComputeSession after advancement.
embedding_vectors: (matrix of float) Concatenated embeddings,
  shaped as (batch * beam * token) x sum_channel(embedding_dim[channel]).
num_steps: The batch was unrolled for these many steps.
component: The name of a Component instance, matching the ComponentSpec.name.
num_channels: The number of FixedFeature channels.
T: The datatype to emit.
pad_to_batch: If set, the op will pad/truncate to this number of elements.
pad_to_steps: If set, the op will pad/truncate to this number of steps.
)doc");

REGISTER_OP("BulkEmbedFixedFeatures")
    .Input("handle: string")
    .Input("embedding_matrix: num_channels * float")
    .Output("output_handle: string")
    .Output("embedding_vectors: float")
    .Output("num_steps: int32")
    .Attr("component: string")
    .Attr("num_channels: int")
    .Attr("pad_to_batch: int")
    .Attr("pad_to_steps: int")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int num_channels;
      TF_RETURN_IF_ERROR(context->GetAttr("num_channels", &num_channels));
      for (int i = 1; i <= num_channels; ++i) {
        TF_RETURN_IF_ERROR(MatrixInputShape(i, context));
      }
      MatrixOutputShape(1, context);
      ScalarOutputShape(2, context);
      return ComputeSessionHandleInputAndOutputShape(context);
    })
    .Doc(R"doc(
This op is a more efficient version of BulkFixedFeatures.

It is intended to be run with large batch sizes at inference time. The op takes
a handle to ComputeSession and embedding matrices as tensor inputs, and directly
outputs concatenated embedding vectors. It calls the BulkEmbedFixedFeatures
method on the underlying component directly, so it requires a padding vector
to be passed.

handle: A handle to ComputeSession.
embedding_matrix: Embedding matrices.
output_handle: A handle to the same ComputeSession after advancement.
embedding_vectors: (matrix of float) Concatenated embeddings,
  shaped as (batch * beam * token) x sum_channel(embedding_dim[channel]).
num_steps: The batch was unrolled for these many steps.
component: The name of a Component instance, matching the ComponentSpec.name.
num_channels: The number of FixedFeature channels.
pad_to_batch: The op will pad/truncate to this number of elements.
pad_to_steps: The op will pad/truncate to this number of steps.
)doc");

REGISTER_OP("BulkEmbedDenseFixedFeatures")
    .Input("handle: string")
    .Input("embedding_matrix: num_channels * float")
    .Output("output_handle: string")
    .Output("embedding_vectors: float")
    .Output("offset_array: int32")
    .Attr("component: string")
    .Attr("num_channels: int")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int num_channels;
      TF_RETURN_IF_ERROR(context->GetAttr("num_channels", &num_channels));
      for (int i = 1; i <= num_channels; ++i) {
        TF_RETURN_IF_ERROR(MatrixInputShape(i, context));
      }
      MatrixOutputShape(1, context);
      VectorOutputShape(2, context);
      return ComputeSessionHandleInputAndOutputShape(context);
    })
    .Doc(R"doc(
This op is a more efficient version of BulkFixedFeatures.

It is intended to be run with large batch sizes at inference time. The op takes
a handle to ComputeSession and embedding matrices as tensor inputs, and directly
outputs concatenated embedding vectors. It calls the BulkEmbedFixedFeatures
method on the underlying component directly, so it requires a padding vector
to be passed.

handle: A handle to ComputeSession.
embedding_matrix: Embedding matrices.
output_handle: A handle to the same ComputeSession after advancement.
embedding_vectors: (matrix of float) Concatenated embeddings, in a dense
array.
offset_array: An array of integers representing the offset of each batch element
in the embedding_vectors array. It is of size (batch+1) and the last element is
the total size of the embedding array.
component: The name of a Component instance, matching the ComponentSpec.name.
num_channels: The number of FixedFeature channels.
)doc");

REGISTER_OP("BulkAdvanceFromOracle")
    .Input("handle: string")
    .Output("output_handle: string")
    .Output("gold_labels: int32")
    .Attr("component: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(1, context);
      return ComputeSessionHandleInputAndOutputShape(context);
    })
    .Doc(R"doc(
Given a ComputeSession, advances until all states are final.

Note that, unlike AdvanceFromOracle, this op does mutate the master state, by
advancing all of its states until they are final.

handle: A handle to a ComputeSession.
output_handle: A handle to the same ComputeSession, after it has advanced.
gold_labels: [batch_size * beam_size * max_num_steps] vector of oracle actions,
             where max_num_steps is the maximum number of steps in the oracle
             action sequences for every state in the batch of beams.  Each
             sub-segment of length max_num_steps provides the oracle action
             sequence for the corresponding state in the batch of beams, padded
             with trailing -1s.
component: The name of a Component instance, matching the ComponentSpec.name.
)doc");

REGISTER_OP("BulkAdvanceFromPrediction")
    .Input("handle: string")
    .Input("scores: T")
    .Output("output_handle: string")
    .Attr("component: string")
    .Attr("T: type")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      TF_RETURN_IF_ERROR(MatrixInputShape(1, context));
      return ComputeSessionHandleInputAndOutputShape(context);
    })
    .Doc(R"doc(
Given a ComputeSession and a tensor of scores, advances the state.

The state will be advanced until all scores are used up or all states are final.

handle: A handle to a ComputeSession.
scores: A tensor of scores with shape
        {batch_size * beam_size * num_steps, num_actions}.
output_handle: handle to the same ComputeSession after advancement.
component: The name of a Component instance, matching the ComponentSpec.name.
T: The datatype to emit.
)doc");

}  // namespace dragnn
}  // namespace syntaxnet
