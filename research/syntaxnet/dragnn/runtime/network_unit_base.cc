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

#include "dragnn/runtime/network_unit_base.h"

#include <string.h>

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns the sum of the dimensions of all channels in the |manager|.  The
// EmbeddingManager template type should be either FixedEmbeddingManager or
// LinkedEmbeddingManager; note that both share the same API.
template <class EmbeddingManager>
size_t SumEmbeddingDimensions(const EmbeddingManager &manager) {
  size_t sum = 0;
  for (size_t i = 0; i < manager.num_channels(); ++i) {
    sum += manager.embedding_dim(i);
  }
  return sum;
}

// Copies each channel of the |embeddings| into the region starting at |data|.
// Returns a pointer to one past the last element of the copied region.  The
// Embeddings type should be FixedEmbeddings or LinkedEmbeddings; note that both
// have the same API.
//
// TODO(googleuser): Try a vectorized copy instead of memcpy().  Unclear whether
// we can do better, though.  For one, the memcpy() implementation may already
// be vectorized.  Also, while the input embeddings are aligned, the output is
// not; e.g., consider concatenating inputs with dims 7 and 9.  This could be
// addressed by requiring that embedding dims are aligned, or by handling the
// unaligned prefix separately.
//
// TODO(googleuser): Consider alternatives for handling fixed feature channels
// with size>1.  The least surprising approach is to concatenate the size>1
// embeddings inside FixedEmbeddings, so the channel IDs still correspond to
// positions in the ComponentSpec.fixed_feature list.  However, that means the
// same embedding gets copied twice, once there and once here.  Conversely, we
// could split the size>1 embeddings into separate channels, eliding a copy
// while obfuscating the channel IDs.  IMO, separate channels seem better
// because very few bits of DRAGNN actually access individual channels, and I
// wrote many of those bits.
template <class Embeddings>
float *CopyEmbeddings(const Embeddings &embeddings, float *data) {
  for (size_t i = 0; i < embeddings.num_embeddings(); ++i) {
    const Vector<float> vector = embeddings.embedding(i);
    memcpy(data, vector.data(), vector.size() * sizeof(float));
    data += vector.size();
  }
  return data;
}

}  // namespace

tensorflow::Status NetworkUnitBase::InitializeBase(
    bool use_concatenated_input, const ComponentSpec &component_spec,
    VariableStore *variable_store, NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  use_concatenated_input_ = use_concatenated_input;
  num_actions_ = component_spec.num_actions();

  TF_RETURN_IF_ERROR(fixed_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));
  TF_RETURN_IF_ERROR(linked_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));
  concatenated_input_dim_ = SumEmbeddingDimensions(fixed_embedding_manager_) +
                            SumEmbeddingDimensions(linked_embedding_manager_);

  if (use_concatenated_input_) {
    // If there is <= 1 input embedding, then the concatenation is trivial and
    // we don't need a local vector; see ConcatenateInput().
    const size_t num_embeddings = fixed_embedding_manager_.num_embeddings() +
                                  linked_embedding_manager_.num_embeddings();
    if (num_embeddings > 1) {
      TF_RETURN_IF_ERROR(network_state_manager->AddLocal(
          concatenated_input_dim_, &concatenated_input_handle_));
    }

    // Check that all fixed features are embedded.
    for (size_t i = 0; i < fixed_embedding_manager_.num_channels(); ++i) {
      if (!fixed_embedding_manager_.is_embedded(i)) {
        return tensorflow::errors::InvalidArgument(
            "Non-embedded fixed features cannot be concatenated");
      }
    }
  }

  extension_manager->GetShared(&fixed_embeddings_handle_);
  extension_manager->GetShared(&linked_embeddings_handle_);
  return tensorflow::Status::OK();
}

tensorflow::Status NetworkUnitBase::EvaluateBase(
    SessionState *session_state, ComputeSession *compute_session,
    Vector<float> *concatenated_input) const {
  FixedEmbeddings &fixed_embeddings =
      session_state->extensions.Get(fixed_embeddings_handle_);
  LinkedEmbeddings &linked_embeddings =
      session_state->extensions.Get(linked_embeddings_handle_);

  TF_RETURN_IF_ERROR(fixed_embeddings.Reset(&fixed_embedding_manager_,
                                            session_state->network_states,
                                            compute_session));
  TF_RETURN_IF_ERROR(linked_embeddings.Reset(&linked_embedding_manager_,
                                             session_state->network_states,
                                             compute_session));

  if (use_concatenated_input_ && concatenated_input != nullptr) {
    *concatenated_input = ConcatenateInput(session_state);
  }
  return tensorflow::Status::OK();
}

Vector<float> NetworkUnitBase::ConcatenateInput(
    SessionState *session_state) const {
  DCHECK(use_concatenated_input_);
  const FixedEmbeddings &fixed_embeddings =
      session_state->extensions.Get(fixed_embeddings_handle_);
  const LinkedEmbeddings &linked_embeddings =
      session_state->extensions.Get(linked_embeddings_handle_);
  const size_t num_embeddings =
      fixed_embeddings.num_embeddings() + linked_embeddings.num_embeddings();

  // Special cases where no actual concatenation is required.
  if (num_embeddings == 0) return {};
  if (num_embeddings == 1) {
    return fixed_embeddings.num_embeddings() > 0
               ? fixed_embeddings.embedding(0)
               : linked_embeddings.embedding(0);
  }

  // General case; concatenate into a local vector.  The ordering of embeddings
  // must be exactly the same as in the Python codebase, which is:
  //   1. Fixed embeddings before linked embeddings (see get_input_tensor() in
  //      network_units.py).
  //   2. In each type, ordered as listed in ComponentSpec.fixed/linked_feature
  //      (see DynamicComponentBuilder._feedforward_unit() in component.py).
  //
  // Since FixedEmbeddings and LinkedEmbeddings already follow the order defined
  // in the ComponentSpec, it suffices to append each fixed embedding, then each
  // linked embedding.
  const MutableVector<float> concatenation =
      session_state->network_states.GetLocal(concatenated_input_handle_);
  float *data = concatenation.data();
  data = CopyEmbeddings(fixed_embeddings, data);
  data = CopyEmbeddings(linked_embeddings, data);
  DCHECK_EQ(data, concatenation.end());

  return Vector<float>(concatenation);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
