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

#include <stddef.h>
#include <set>

#include "dragnn/runtime/session_state.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Maximum number of free states.
static constexpr size_t kMaxFreeStates = 16;

class SessionStatePoolTest : public ::testing::Test {
 protected:
  SessionStatePool pool_{kMaxFreeStates};
};

// Tests that ScopedSessionState can be used to acquire a valid state.
TEST_F(SessionStatePoolTest, ScopedWrapper) {
  const ScopedSessionState state(&pool_);
  EXPECT_TRUE(state.get());  // non-null
}

// Tests that the active states claimed from the pool are unique.
TEST_F(SessionStatePoolTest, UniqueActiveStates) {
  // NB: Don't use std::unique_ptr<ScopedSessionState> in real code.  The test
  // does this because it's otherwise difficult to acquire lots of states.
  std::vector<std::unique_ptr<ScopedSessionState>> states;
  for (size_t i = 0; i < 100; ++i) {
    states.emplace_back(new ScopedSessionState(&pool_));
  }

  // Check that all of the states are unique.
  std::set<const SessionState *> state_ptrs;
  for (const auto &state : states) {
    EXPECT_TRUE(state_ptrs.insert(state->get()).second);
  }
  EXPECT_TRUE(state_ptrs.find(nullptr) == state_ptrs.end());
}

// Tests that active states, when released, are reclaimed and reused.
TEST_F(SessionStatePoolTest, Reuse) {
  std::set<const SessionState *> state_ptrs;

  {  // Grab exactly as many states as the free list can hold.
    std::vector<std::unique_ptr<ScopedSessionState>> states;
    for (size_t i = 0; i < kMaxFreeStates; ++i) {
      states.emplace_back(new ScopedSessionState(&pool_));
      EXPECT_TRUE(state_ptrs.insert(states.back()->get()).second);
    }
  }

  {  // Grab the same number of states again and check that they are the same
     // objects we saw in the first loop.
    std::vector<std::unique_ptr<ScopedSessionState>> states;
    for (size_t i = 0; i < kMaxFreeStates; ++i) {
      states.emplace_back(new ScopedSessionState(&pool_));
      EXPECT_FALSE(state_ptrs.insert(states.back()->get()).second);
    }
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
