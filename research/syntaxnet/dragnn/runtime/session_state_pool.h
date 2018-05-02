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

#ifndef DRAGNN_RUNTIME_SESSION_STATE_POOL_H_
#define DRAGNN_RUNTIME_SESSION_STATE_POOL_H_

#include <stddef.h>
#include <memory>
#include <utility>

#include "dragnn/runtime/session_state.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A thread-safe pool of session states that maintains a free list.  The free
// list is bounded, so a spike in usage does not permanently increase the size
// of the pool.  Use ScopedSessionState to interact with the pool.
class SessionStatePool {
 public:
  // Creates a pool whose free list holds at most |max_free_states| states.
  //
  // If usage spikes are not a concern (e.g., during offline processing where
  // the runtime is called from a fixed-size pool of threads), then specify a
  // large value like SIZE_MAX.  That eliminates unnecessary deallocations and
  // reallocations, and eliminates the need to coordinate the thread pool size
  // with this pool's size.
  //
  // If memory usage dominates CPU usage, then specify 0 to eliminate overhead
  // from the free list.
  //
  // TODO(googleuser): An alternative is to set a target allocation
  // rate (e.g., 2% of Acquire()s should create a new state), and let the pool
  // adapt its free list size to achieve that rate.
  explicit SessionStatePool(size_t max_free_states);

 private:
  friend class ScopedSessionState;

  // Returns a state acquired from this pool.  The caller is the exclusive user
  // of the returned state until it is passed to Release().
  std::unique_ptr<SessionState> Acquire();

  // Releases the |state| back to this pool.  The |state| must be the result of
  // a previous Acquire().  The caller can no longer use the |state|.
  void Release(std::unique_ptr<SessionState> state);

  // Maximum number of states to keep in the |free_list_|.
  const size_t max_free_states_;

  // Mutex guarding the |free_list_|.
  tensorflow::mutex mutex_;

  // List of previously-Release()d states.
  std::vector<std::unique_ptr<SessionState>> free_list_ GUARDED_BY(mutex_);
};

// RAII wrapper that manages a session state acquired from a pool.  The wrapped
// state is usable during the lifetime of the wrapper.
class ScopedSessionState {
 public:
  // Implements RAII semantics.
  explicit ScopedSessionState(SessionStatePool *pool)
      : pool_(pool), state_(pool_->Acquire()) {}
  ~ScopedSessionState() { pool_->Release(std::move(state_)); }

  // Prevents double-release.
  ScopedSessionState(const ScopedSessionState &that) = delete;
  ScopedSessionState &operator=(const ScopedSessionState &that) = delete;

  // Provides std::unique_ptr-like access.
  SessionState *get() const { return state_.get(); }
  SessionState &operator*() const { return *get(); }
  SessionState *operator->() const { return get(); }

 private:
  // Pool from which the |state_| was acquired.
  SessionStatePool *const pool_;

  // Wrapped session state.
  std::unique_ptr<SessionState> state_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_SESSION_STATE_POOL_H_
