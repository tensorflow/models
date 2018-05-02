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

// Utils for configuring and extracting linked embeddings for sequence-based
// models.  Analogous to LinkedEmbeddingManager and LinkedEmbeddings, but uses
// SequenceLinker instead of ComputeSession.

#ifndef DRAGNN_RUNTIME_SEQUENCE_LINKS_H_
#define DRAGNN_RUNTIME_SEQUENCE_LINKS_H_

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_linker.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Manager for linked embeddings for sequence-based models.  This is a wrapper
// around the LinkedEmbeddingManager.
class SequenceLinkManager {
 public:
  // Creates an empty manager.
  SequenceLinkManager() = default;

  // Resets this to wrap the |linked_embedding_manager|, which must outlive
  // this.  The |sequence_linker_types| should name one SequenceLinker subclass
  // per channel; e.g., {"IdentitySequenceLinker", "ReversedSequenceLinker"}.
  // This initializes each SequenceLinker from the |component_spec|.  On error,
  // returns non-OK and does not modify this.
  tensorflow::Status Reset(
      const LinkedEmbeddingManager *linked_embedding_manager,
      const ComponentSpec &component_spec,
      const std::vector<string> &sequence_linker_types);

  // Accessors.
  size_t num_channels() const { return channel_configs_.size(); }

 private:
  friend class SequenceLinks;

  // Configuration for a single linked embedding channel.
  struct ChannelConfig {
    // Whether this link is recurrent.
    bool is_recurrent = false;

    // Handle to the source layer in the relevant NetworkStates.
    LayerHandle<float> handle;

    // Extractor for sequences of translated link indices.
    std::unique_ptr<SequenceLinker> linker;
  };

  // Array of zeros that can be substituted for out-of-bounds embeddings.  This
  // is a reference to the corresponding array in the LinkedEmbeddingManager.
  // See the large comment in linked_embeddings.cc for reference.
  AlignedView zeros_;

  // Ordered list of configurations for each channel.
  std::vector<ChannelConfig> channel_configs_;
};

// A set of linked embeddings for a sequence-based model.  Configured by a
// SequenceLinkManager.
class SequenceLinks {
 public:
  // Creates an empty set of embeddings.
  SequenceLinks() = default;

  // Resets this to the sequences of linked embeddings managed by the |manager|
  // on the |input|.  Retrieves layers from the |network_states|.  The |manager|
  // must live until this is destroyed or Reset(), and should not be modified
  // during that time.  If |add_steps| is true, then infers the number of steps
  // from the non-recurrent links and adds steps to the |network_states| before
  // processing the recurrent links.  On error, returns non-OK.
  //
  // NB: Recurrent links are tricky, because the |network_states| must be filled
  // with steps before processing recurrent links.  There are two approaches:
  // 1. Add steps to the |network_states| before calling Reset().  This only
  //    works if the component also has fixed features, which can be used to
  //    infer the number of steps.
  // 2. Set |add_steps| to true, so steps are added during Reset().  This only
  //    works if the component also has non-recurrent links, which can be used
  //    to infer the number of steps.
  // If a component only has recurrent links then neither of the above works,
  // but such a component would be nonsensical: it recurses on itself with no
  // external input.
  tensorflow::Status Reset(bool add_steps, const SequenceLinkManager *manager,
                           NetworkStates *network_states,
                           InputBatchCache *input);

  // Retrieves the linked embedding for the |target_index|'th element of the
  // |channel_id|'th channel.  Sets |embedding| to the linked embedding vector
  // and sets |is_out_of_bounds| to true if the link is out of bounds.
  void Get(size_t channel_id, size_t target_index, Vector<float> *embedding,
           bool *is_out_of_bounds) const;

  // Accessors.
  size_t num_channels() const { return num_channels_; }
  size_t num_steps() const { return num_steps_; }

 private:
  // Data associated with a single linked embedding channel.
  struct Channel {
    // Source layer activations.
    Matrix<float> layer;

    // Translated link indices for each step.
    std::vector<int32> links;
  };

  // Zero vector from the most recent Reset().
  AlignedView zeros_;

  // Number of channels and steps from the most recent Reset().
  size_t num_channels_ = 0;
  size_t num_steps_ = 0;

  // Ordered list of linked embedding channels.  This may contain more than
  // |num_channels_| entries, to avoid deallocation/reallocation cycles, but
  // only the first |num_channels_| entries are valid.
  std::vector<Channel> channels_;
};

// Implementation details below.

inline void SequenceLinks::Get(size_t channel_id, size_t target_index,
                               Vector<float> *embedding,
                               bool *is_out_of_bounds) const {
  DCHECK_LT(channel_id, num_channels());
  DCHECK_LT(target_index, num_steps());
  const Channel &channel = channels_[channel_id];
  const int32 link = channel.links[target_index];
  *is_out_of_bounds = (link < 0 || link >= channel.layer.num_rows());
  if (*is_out_of_bounds) {
    *embedding = Vector<float>(zeros_, channel.layer.num_columns());
  } else {
    *embedding = channel.layer.row(link);
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_SEQUENCE_LINKS_H_
