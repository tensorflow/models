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

#include "dragnn/runtime/sequence_backend.h"

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that the "reverse-*" step lookup functions ignore the batch and beam
// indices and return -1 if the sequence size was never set.
TEST(SequenceBackendTest, ReverseCharUninitialized) {
  for (const string &reverse_method : {"reverse-char", "reverse-token"}) {
    SequenceBackend backend;
    const std::function<int(int, int, int)> reverse =
        backend.GetStepLookupFunction(reverse_method);

    EXPECT_EQ(reverse(0, 0, 0), -1);
    EXPECT_EQ(reverse(1, 1, 1), -1);
    EXPECT_EQ(reverse(-1, -1, -1), -1);
    EXPECT_EQ(reverse(0, 0, 9999), -1);
    EXPECT_EQ(reverse(0, 0, -9999), -1);
  }
}

// Tests that the "reverse-*" step lookup functions ignore the batch and beam
// indices and return the reverse of the step index w.r.t. the most recent call
// to SetSequenceSize().
TEST(SequenceBackendTest, ReverseCharAfterSetSequenceSize) {
  for (const string &reverse_method : {"reverse-char", "reverse-token"}) {
    SequenceBackend backend;
    const std::function<int(int, int, int)> reverse =
        backend.GetStepLookupFunction(reverse_method);

    backend.SetSequenceSize(10);
    EXPECT_EQ(reverse(0, 0, -1), -1);
    EXPECT_EQ(reverse(0, 0, 0), 9);
    EXPECT_EQ(reverse(1, 1, 1), 8);
    EXPECT_EQ(reverse(8, 8, 8), 1);
    EXPECT_EQ(reverse(9, 9, 9), 0);
    EXPECT_EQ(reverse(10, 10, 10), -1);
    EXPECT_EQ(reverse(-1, -1, 5), 4);
    EXPECT_EQ(reverse(0, 0, 9999), -1);
    EXPECT_EQ(reverse(0, 0, -9999), -1);

    backend.SetSequenceSize(11);
    EXPECT_EQ(reverse(0, 0, -1), -1);
    EXPECT_EQ(reverse(0, 0, 0), 10);
    EXPECT_EQ(reverse(1, 1, 1), 9);
    EXPECT_EQ(reverse(8, 8, 8), 2);
    EXPECT_EQ(reverse(9, 9, 9), 1);
    EXPECT_EQ(reverse(10, 10, 10), 0);
    EXPECT_EQ(reverse(-1, -1, 5), 5);
    EXPECT_EQ(reverse(0, 0, 9999), -1);
    EXPECT_EQ(reverse(0, 0, -9999), -1);
  }
}

// Tests that the input beam is forwarded.
TEST(SequenceBackendTest, BeamForwarding) {
  SequenceBackend backend;

  const TransitionState *parent_state = nullptr;
  parent_state += 1234;  // arbitrary non-null pointer
  const std::vector<std::vector<const TransitionState *>> parent_states = {
      {parent_state}};
  const int ignored_max_beam_size = 999;
  InputBatchCache *ignored_input = nullptr;
  backend.InitializeData(parent_states, ignored_max_beam_size, ignored_input);

  EXPECT_EQ(backend.GetBeam(), parent_states);
}

// Tests the accessors of the backend.
TEST(SequenceBackendTest, Accessors) {
  SequenceBackend backend;

  ComponentSpec spec;
  spec.set_name("foo");
  backend.InitializeComponent(spec);

  EXPECT_EQ(backend.Name(), "foo");
  EXPECT_EQ(backend.BeamSize(), 1);
  EXPECT_EQ(backend.BatchSize(), 1);
  EXPECT_TRUE(backend.IsReady());
  EXPECT_TRUE(backend.IsTerminal());
}

// Tests the trivial mutators of the backend.
TEST(SequenceBackendTest, Mutators) {
  SequenceBackend backend;

  // These are NOPs and should not crash.
  backend.FinalizeData();
  backend.ResetComponent();
  backend.InitializeTracing();
  backend.DisableTracing();
}

// Tests the beam index accessors of the backend.
TEST(SequenceBackendTest, BeamIndex) {
  SequenceBackend backend;

  // This always returns the current_index (first arg).
  EXPECT_EQ(backend.GetSourceBeamIndex(0, 0), 0);
  EXPECT_EQ(backend.GetSourceBeamIndex(1, 2), 1);
  EXPECT_EQ(backend.GetSourceBeamIndex(-1, -1), -1);
  EXPECT_EQ(backend.GetSourceBeamIndex(10, 99), 10);

  // This always returns 0.
  EXPECT_EQ(backend.GetBeamIndexAtStep(0, 0, 0), 0);
  EXPECT_EQ(backend.GetBeamIndexAtStep(1, 2, 3), 0);
  EXPECT_EQ(backend.GetBeamIndexAtStep(-1, -1, -1), 0);
  EXPECT_EQ(backend.GetBeamIndexAtStep(123, 456, 789), 0);
}

// Tests the that the backend produces a single empty trace.
TEST(SequenceBackendTest, Tracing) {
  SequenceBackend backend;

  const ComponentTrace empty_trace;
  const auto actual_traces = backend.GetTraceProtos();
  ASSERT_EQ(actual_traces.size(), 1);
  ASSERT_EQ(actual_traces[0].size(), 1);
  EXPECT_THAT(actual_traces[0][0], test::EqualsProto(empty_trace));
}

// Tests the unsupported methods of the backend.
TEST(SequenceBackendTest, UnsupportedMethods) {
  SequenceBackend backend;

  EXPECT_DEATH(backend.StepsTaken(0), "Not supported");
  EXPECT_DEATH(backend.AdvanceFromPrediction(nullptr, 0, 0), "Not supported");
  EXPECT_DEATH(backend.AdvanceFromOracle(), "Not supported");
  EXPECT_DEATH(backend.GetOracleLabels(), "Not supported");
  EXPECT_DEATH(backend.GetFixedFeatures(nullptr, nullptr, nullptr, 0),
               "Not supported");
  EXPECT_DEATH(backend.BulkGetFixedFeatures(
                   BulkFeatureExtractor(nullptr, nullptr, nullptr)),
               "Not supported");
  EXPECT_DEATH(backend.BulkEmbedFixedFeatures(0, 0, 0, {}, nullptr),
               "Not supported");
  EXPECT_DEATH(backend.GetRawLinkFeatures(0), "Not supported");
  EXPECT_DEATH(backend.AddTranslatedLinkFeaturesToTrace({}, 0),
               "Not supported");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
