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

#include "dragnn/runtime/session_state_pool.h"

#include <algorithm>

namespace syntaxnet {
namespace dragnn {
namespace runtime {

SessionStatePool::SessionStatePool(size_t max_free_states)
    : max_free_states_(max_free_states) {}

std::unique_ptr<SessionState> SessionStatePool::Acquire() {
  {  // Exclude the slow path from the critical region.
    tensorflow::mutex_lock lock(mutex_);
    if (!free_list_.empty()) {
      // Fast path: reuse a free state.
      std::unique_ptr<SessionState> state = std::move(free_list_.back());
      free_list_.pop_back();
      return state;
    }
  }

  // Slow path: allocate a new state.
  return std::unique_ptr<SessionState>(new SessionState());
}

void SessionStatePool::Release(std::unique_ptr<SessionState> state) {
  {  // Exclude the slow path from the critical region.
    tensorflow::mutex_lock lock(mutex_);
    if (free_list_.size() < max_free_states_) {
      // Fast path: reclaim in the free list.
      free_list_.emplace_back(std::move(state));
      return;
    }
  }

  // Slow path: discard the excess |state| when it goes out of scope.
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
