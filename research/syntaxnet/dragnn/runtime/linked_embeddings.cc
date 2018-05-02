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

#include "dragnn/runtime/linked_embeddings.h"

#include <string.h>
#include <algorithm>
#include <utility>

#include "dragnn/protos/data.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/arithmetic.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns the name of the weight matrix for the |channel_id|'th linked feature
// channel of the |component_spec|.
string LinkedWeightMatrixVariableName(const ComponentSpec &component_spec,
                                      int channel_id) {
  // Cf. _add_hooks_for_linked_embedding_matrix() in runtime_support.py.
  return tensorflow::strings::StrCat(component_spec.name(),
                                     "/linked_embedding_matrix_", channel_id,
                                     "/weights");
}

// As above, but for the out-of-bounds vector.
string LinkedOutOfBoundsVectorVariableName(const ComponentSpec &component_spec,
                                           int channel_id) {
  // Cf. _add_hooks_for_linked_embedding_matrix() in runtime_support.py.
  return tensorflow::strings::StrCat(component_spec.name(),
                                     "/linked_embedding_matrix_", channel_id,
                                     "/out_of_bounds");
}

}  // namespace

tensorflow::Status LinkedEmbeddingManager::Reset(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager) {
  const int num_channels = component_spec.linked_feature_size();
  std::vector<ChannelConfig> channel_configs(num_channels);
  size_t zeros_dimension = 0;  // required dimension for the shared zero vector
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    const LinkedFeatureChannel &channel_spec =
        component_spec.linked_feature(channel_id);
    ChannelConfig &channel_config = channel_configs[channel_id];

    if (channel_spec.size() < 1) {
      return tensorflow::errors::InvalidArgument(
          "Invalid channel size for channel ", channel_id, ": ",
          channel_spec.ShortDebugString());
    }

    if (channel_spec.size() > 1) {
      return tensorflow::errors::Unimplemented(
          "Multi-instance linked features are not supported for channel ",
          channel_id, ": ", channel_spec.ShortDebugString());
    }

    size_t source_dimension = 0;
    TF_RETURN_IF_ERROR(network_state_manager->LookupLayer(
        channel_spec.source_component(), channel_spec.source_layer(),
        &source_dimension, &channel_config.source_handle));

    channel_config.is_transformed = channel_spec.embedding_dim() >= 0;
    if (!channel_config.is_transformed) {
      // Out-of-bounds direct links may be pointed at |zeros_|, so it must be
      // large enough for any direct link.
      channel_config.dimension = source_dimension;
      zeros_dimension = std::max(zeros_dimension, channel_config.dimension);
      continue;
    }

    // The remainder of this loop initializes transformed links.
    channel_config.dimension = channel_spec.embedding_dim();
    TF_RETURN_IF_ERROR(network_state_manager->AddLocal(
        channel_config.dimension, &channel_config.product_handle));

    const string debug_name = tensorflow::strings::StrCat(
        component_spec.name(), ".", channel_spec.name());
    TF_RETURN_IF_ERROR(channel_config.weight_matrix.Initialize(
        debug_name, LinkedWeightMatrixVariableName(component_spec, channel_id),
        channel_spec.embedding_dim(), variable_store));
    const FlexibleMatrixKernel &weights = channel_config.weight_matrix;

    Vector<float> &out_of_bounds_vector = channel_config.out_of_bounds_vector;
    TF_RETURN_IF_ERROR(variable_store->Lookup(
        LinkedOutOfBoundsVectorVariableName(component_spec, channel_id),
        &out_of_bounds_vector));

    if (weights.NumColumns() != source_dimension) {
      return tensorflow::errors::InvalidArgument(
          "Weight matrix does not match source layer in link ", channel_id,
          ": weights=[", weights.NumPaddedRows(), ", ", weights.NumColumns(),
          "] vs layer_dim=", source_dimension);
    }

    if (!weights.MatchesOutputDimension(channel_config.dimension)) {
      return tensorflow::errors::InvalidArgument(
          "Weight matrix shape should be output dimension plus padding. ",
          "Linked channel ID: ", channel_id, ": weights=[",
          weights.NumPaddedRows(), ", ", weights.NumColumns(),
          "] vs output=", channel_config.dimension);
    }

    if (out_of_bounds_vector.size() != channel_config.dimension) {
      return tensorflow::errors::InvalidArgument(
          "Out-of-bounds vector does not match embedding_dim in link ",
          channel_id, ": out_of_bounds=[", out_of_bounds_vector.size(),
          "] vs embedding_dim=", channel_config.dimension);
    }
  }

  // Success; make modifications.
  component_name_ = component_spec.name();
  channel_configs_ = std::move(channel_configs);
  zeros_.Resize(zeros_dimension * sizeof(float));
  memset(zeros_.view().data(), 0, zeros_.view().size());
  return tensorflow::Status::OK();
}

tensorflow::Status LinkedEmbeddings::Reset(
    const LinkedEmbeddingManager *manager, const NetworkStates &network_states,
    ComputeSession *compute_session) {
  const int num_channels = manager->channel_configs_.size();
  channels_.resize(num_channels);
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    Channel &channel = channels_[channel_id];
    const std::vector<LinkFeatures> features =
        compute_session->GetTranslatedLinkFeatures(manager->component_name(),
                                                   channel_id);

    // Since we require LinkedFeatureChannel.size==1, there should be exactly
    // one linked feature.
    if (features.size() != 1) {
      return tensorflow::errors::Internal(
          "Got ", features.size(), " linked features; expected 1 for channel ",
          channel_id);
    }
    const LinkFeatures &feature = features[0];

    if (feature.batch_idx() > 0) {
      return tensorflow::errors::Unimplemented(
          "Batches are not supported for channel ", channel_id);
    }

    if (feature.beam_idx() > 0) {
      return tensorflow::errors::Unimplemented(
          "Beams are not supported for channel ", channel_id);
    }

    const int source_beam_size = compute_session->SourceComponentBeamSize(
        manager->component_name(), channel_id);
    if (source_beam_size != 1) {
      return tensorflow::errors::Unimplemented(
          "Source beams are not supported for channel ", channel_id);
    }

    // Consider these bits of the TF-based DRAGNN codebase:
    //   1. The ExtractLinkFeatures op in dragnn_op_kernels.cc substitutes -1
    //      for missing step indices, and clips all step indices to a min of -1.
    //   2. activation_lookup_*() in network_units.py adds +1 to step indices.
    //   3. Layer.create_array() in network_units.py starts each TensorArray
    //      with a zero vector.
    // Therefore, a direct link with a missing or negative step index should
    // receive a zeroed embedding.  Regarding transformed links:
    //   4. NetworkUnitInterface.__init__() in network_units.py extends the
    //      linked embedding matrix by 1 row.
    //   5. pass_through_embedding_matrix() in network_units.py extends each
    //      input activation vector with a 0/1 out-of-bounds indicator.
    // The result of multiplying the extended linked embedding matrix with the
    // extended input activation vector is:
    //   * If in-bounds: The product of the non-extended matrix and vector.
    //   * If out-of-bounds: The last row of the extended matrix.
    const bool is_out_of_bounds =
        !feature.has_step_idx() || feature.step_idx() < 0;
    channel.is_out_of_bounds = is_out_of_bounds;

    const LinkedEmbeddingManager::ChannelConfig &channel_config =
        manager->channel_configs_[channel_id];
    if (is_out_of_bounds) {
      if (channel_config.is_transformed) {
        // Point at the special out-of-bounds embedding.
        channel.embedding = channel_config.out_of_bounds_vector;
      } else {
        // Point at a prefix of the zero vector.
        //
        // TODO(googleuser): Consider providing is_zero(channel_id)
        // so we can elide ops on zero vectors later on in the pipeline.  This
        // would help if out-of-bounds links are frequent.
        channel.embedding =
            Vector<float>(manager->zeros_.view(), channel_config.dimension);
      }
    } else {
      // Point at the activation vector of the translated step index.
      channel.embedding = network_states.GetLayer(channel_config.source_handle)
                              .row(feature.step_idx());
      if (channel_config.is_transformed) {
        // Multiply with the weight matrix and point at the result.
        const MutableVector<float> product =
            network_states.GetLocal(channel_config.product_handle);

        channel_config.weight_matrix.MatrixVectorProduct(channel.embedding,
                                                         product);
        channel.embedding = product;
      }
    }

    DCHECK_EQ(channel.embedding.size(), channel_config.dimension);
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
