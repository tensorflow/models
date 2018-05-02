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

#include "dragnn/runtime/fixed_embeddings.h"

#include <string.h>
#include <algorithm>
#include <limits>
#include <utility>

#include "dragnn/runtime/math/arithmetic.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns the name of the embedding matrix for the |channel_id|'th fixed
// feature channel of the |component_spec|.
string FixedEmbeddingMatrixVariableName(const ComponentSpec &component_spec,
                                        int channel_id) {
  // Cf. _add_hooks_for_fixed_embedding_matrix() in runtime_support.py.
  return tensorflow::strings::StrCat(component_spec.name(),
                                     "/fixed_embedding_matrix_", channel_id,
                                     "/trimmed");
}

// Resizes |buffer| to |size| and returns the array it manages.  Helper for the
// allocator functors used by ComputeSession::GetInputFeatures().
template <class T>
T *Alloc(int size, std::vector<T> *buffer) {
  buffer->resize(size);
  return buffer->data();
}

// Returns true if two pointers have the same address.
bool SameAddress(const void *pointer1, const void *pointer2) {
  return pointer1 == pointer2;
}

// Number of IDs to allow per embedding.
constexpr size_t kMaxNumFeatureIds = 1;

}  // namespace

tensorflow::Status FixedEmbeddingManager::Reset(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager) {
  const int num_channels = component_spec.fixed_feature_size();
  std::vector<ChannelConfig> channel_configs(num_channels);
  size_t max_dimension = 0;  // maximum dimension across all channels
  size_t num_embeddings = 0;
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    const FixedFeatureChannel &channel_spec =
        component_spec.fixed_feature(channel_id);
    ChannelConfig &channel_config = channel_configs[channel_id];

    if (channel_spec.size() < 1) {
      return tensorflow::errors::InvalidArgument(
          "Invalid channel size for channel ", channel_id, ": ",
          channel_spec.ShortDebugString());
    }
    const size_t channel_size = channel_spec.size();
    channel_config.channel_base = num_embeddings;
    num_embeddings += channel_size;
    channel_config.handles.resize(channel_size);
    channel_config.is_embedded = channel_spec.embedding_dim() >= 0;

    // Configure non-embedded channels separately.
    if (!channel_config.is_embedded) {
      for (size_t i = 0; i < channel_size; ++i) {
        TF_RETURN_IF_ERROR(network_state_manager->AddLocal(
            kMaxNumFeatureIds, &channel_config.handles[i].ids));
      }
      continue;
    }

    // The remainder of the loop configures embedded channels.
    const size_t dimension = channel_spec.embedding_dim();
    max_dimension = std::max(max_dimension, dimension);

    for (size_t i = 0; i < channel_size; ++i) {
      TF_RETURN_IF_ERROR(network_state_manager->AddLocal(
          dimension, &channel_config.handles[i].sum));
    }

    Matrix<float> &embedding_matrix = channel_config.embedding_matrix;
    TF_RETURN_IF_ERROR(variable_store->Lookup(
        FixedEmbeddingMatrixVariableName(component_spec, channel_id),
        &embedding_matrix));

    if (embedding_matrix.num_rows() != channel_spec.vocabulary_size()) {
      return tensorflow::errors::InvalidArgument(
          "ComponentSpec (", channel_spec.vocabulary_size(),
          ") and VariableStore (", embedding_matrix.num_rows(),
          ") disagree on vocabulary size for channel ", channel_id, ": ",
          channel_spec.ShortDebugString());
    }

    if (embedding_matrix.num_columns() != dimension) {
      return tensorflow::errors::InvalidArgument(
          "ComponentSpec (", dimension, ") and VariableStore (",
          embedding_matrix.num_columns(),
          ") disagree on embedding dim for channel ", channel_id, ": ",
          channel_spec.ShortDebugString());
    }
  }

  // Success; make modifications.
  component_name_ = component_spec.name();
  num_embeddings_ = num_embeddings;
  channel_configs_ = std::move(channel_configs);
  zeros_.Resize(max_dimension * sizeof(float));
  memset(zeros_.view().data(), 0, zeros_.view().size());
  return tensorflow::Status::OK();
}

tensorflow::Status FixedEmbeddings::Reset(const FixedEmbeddingManager *manager,
                                          const NetworkStates &network_states,
                                          ComputeSession *compute_session) {
  const AlignedView zeros(manager->zeros_.view());
  const size_t num_channels = manager->num_channels();
  features_.clear();
  features_.reserve(manager->num_embeddings());
  for (size_t channel_id = 0; channel_id < num_channels; ++channel_id) {
    const FixedEmbeddingManager::ChannelConfig &channel_config =
        manager->channel_configs_[channel_id];
    const std::vector<FixedEmbeddingManager::Handle> &handles =
        channel_config.handles;
    const size_t channel_base = channel_config.channel_base;
    const size_t channel_size = handles.size();
    DCHECK_EQ(channel_base, features_.size());
    DCHECK_LE(channel_base + channel_size, manager->num_embeddings());

    const int num_features = compute_session->GetInputFeatures(
        manager->component_name(),
        [this](int size) { return Alloc(size, &indices_); },
        [this](int size) { return Alloc(size, &ids_); },
        [this](int size) { return Alloc(size, &weights_); }, channel_id);
    DCHECK_EQ(num_features, indices_.size());
    DCHECK_EQ(num_features, ids_.size());
    DCHECK_EQ(num_features, weights_.size());
    DCHECK(std::all_of(indices_.begin(), indices_.end(),
                       [channel_size](int32 index) {
                         return index >= 0 && index < channel_size;
                       }));

    // Handle non-embedded channels separately.
    if (!channel_config.is_embedded) {
      for (size_t index = 0; index < channel_size; ++index) {
        features_.emplace_back(/*is_embedded=*/false);
        features_.back().ids = network_states.GetLocal(handles[index].ids);
        features_.back().ids[0] = -1;  // so we can check that all IDs are set
      }

      for (int feature = 0; feature < num_features; ++feature) {
        const int32 index = indices_[feature];
        const int64 id = ids_[feature];
        if (id < 0 || id > std::numeric_limits<int32>::max()) {
          return tensorflow::errors::Internal(
              "Component '", manager->component_name_, "' channel ", channel_id,
              " index ", index, ": Invalid non-embedded feature ID ", id);
        }

        const float weight = weights_[feature];
        if (weight != 1.0) {
          return tensorflow::errors::Internal(
              "Component '", manager->component_name_, "' channel ", channel_id,
              " index ", index, ": Invalid non-embedded feature weight ",
              weight, " (expected 1.0)");
        }

        int32 &output_id = features_[channel_base + index].ids[0];
        if (output_id != -1) {
          return tensorflow::errors::Internal(
              "Component '", manager->component_name_, "' channel ", channel_id,
              " index ", index, ": Duplicate non-embedded feature ID ", id);
        }

        output_id = id;
      }

      for (size_t index = 0; index < channel_size; ++index) {
        if (features_[channel_base + index].ids[0] == -1) {
          return tensorflow::errors::Internal(
              "Component '", manager->component_name_, "' channel ", channel_id,
              " index ", index, ": Missing non-embedded feature ID");
        }
      }

      continue;
    }

    // The remainder of the loop handles embedded channels.
    const Matrix<float> &embedding_matrix = channel_config.embedding_matrix;

    // Acquire the local sum operands and initialize embeddings to zero.
    sums_.resize(channel_size);
    for (size_t i = 0; i < channel_size; ++i) {
      sums_[i] = network_states.GetLocal(handles[i].sum);
      features_.emplace_back(/*is_embedded=*/true);
      features_.back().embedding = Vector<float>(zeros, sums_[i].size());
    }

    // Add in a weighted embedding for each feature.  The extracted features do
    // not have any ordering guarantee (e.g., sorted by |indices|), which makes
    // applying special-case shortcuts difficult, but not impossible.  If the
    // features did have an ordering guarantee, we could use a less intricate
    // algorithm, but it's not clear if it would be much faster.
    for (int feature = 0; feature < num_features; ++feature) {
      const int32 index = indices_[feature];
      const int64 id = ids_[feature];
      const float weight = weights_[feature];
      const Vector<float> row = embedding_matrix.row(id);
      const MutableVector<float> sum = sums_[index];
      Vector<float> &embedding = features_[channel_base + index].embedding;

      if (SameAddress(embedding.data(), zeros.data())) {
        // If the |embedding| points at |zeros|, then this is the first addition
        // so we can use simplified arithmetic.
        if (weight == 1.0) {
          // Trivial scaling: Point at the |row|.
          embedding = row;
        } else {
          // Adding to zero: Scale into the |sum| and point at it.
          ScaleElements(weight, row, sum);
          embedding = sum;
        }
      } else {
        if (!SameAddress(embedding.data(), sum.data())) {
          // If the |embedding| does not point at |zeros| or |sum|, then this is
          // the second addition and we also used the "Trivial scaling" shortcut
          // in the first addition.  Therefore, the |embedding| currently points
          // at another row of the embedding matrix.  Copy that row to |sum| and
          // point at it, so we can add the current row to it.
          memcpy(sum.data(), embedding.data(), sum.size() * sizeof(float));
          embedding = sum;
        }

        // General case: Add to the |sum|, which is aliased by the |embedding|.
        AddScaledElements(weight, row, sum);
      }

      DCHECK_EQ(embedding.size(), embedding_matrix.num_columns());
    }
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
