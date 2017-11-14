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

#include "dragnn/core/component_registry.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_transition_state.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/protos/data.pb.h"
#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace {

const char kSentence0[] = R"(
token {
  word: "Sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
  break_level: NO_BREAK
}
token {
  word: "0" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "." start: 10 end: 10 head: 0 tag: "." category: "." label: "punct"
  break_level: NO_BREAK
}
)";

const char kSentence1[] = R"(
token {
  word: "Sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
  break_level: NO_BREAK
}
token {
  word: "1" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "." start: 10 end: 10 head: 0 tag: "." category: "." label: "punct"
  break_level: NO_BREAK
}
)";

const char kLongSentence[] = R"(
token {
  word: "Sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
  break_level: NO_BREAK
}
token {
  word: "1" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "2" start: 10 end: 10 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "3" start: 11 end: 11 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "." start: 12 end: 12 head: 0 tag: "." category: "." label: "punct"
  break_level: NO_BREAK
}
)";

const char kMasterSpec[] = R"(
component {
  name: "test"
  transition_system {
    registered_name: "shift-only"
  }
  linked_feature {
    name: "prev"
    fml: "input.focus"
    embedding_dim: 32
    size: 1
    source_component: "prev"
    source_translator: "identity"
    source_layer: "last_layer"
  }
  backend {
    registered_name: "StatelessComponent"
  }
}
)";

}  // namespace

using testing::Return;

class StatelessComponentTest : public ::testing::Test {
 public:
  std::unique_ptr<Component> CreateParser(
      int beam_size,
      const std::vector<std::vector<const TransitionState *>> &states,
      const std::vector<string> &data) {
    MasterSpec master_spec;
    CHECK(TextFormat::ParseFromString(kMasterSpec, &master_spec));
    data_.reset(new InputBatchCache(data));

    // The stateless component does not use any particular input batch type, and
    // relies on the preceding components to convert the input batch.
    data_->GetAs<SentenceInputBatch>();

    // Create a parser component with the specified beam size.
    std::unique_ptr<Component> parser_component(
        Component::Create("StatelessComponent"));
    parser_component->InitializeComponent(master_spec.component(0));
    parser_component->InitializeData(states, beam_size, data_.get());
    return parser_component;
  }

  std::unique_ptr<InputBatchCache> data_;
};

TEST_F(StatelessComponentTest, ForwardsTransitionStates) {
  MockTransitionState mock_state_1, mock_state_2, mock_state_3;
  const std::vector<std::vector<const TransitionState *>> parent_states = {
      {}, {&mock_state_1}, {&mock_state_2, &mock_state_3}};

  std::vector<string> data;
  for (const string &textproto : {kSentence0, kSentence1, kLongSentence}) {
    Sentence sentence;
    CHECK(TextFormat::ParseFromString(textproto, &sentence));
    data.emplace_back();
    CHECK(sentence.SerializeToString(&data.back()));
  }
  CHECK_EQ(parent_states.size(), data.size());

  const int kBeamSize = 2;
  auto test_parser = CreateParser(kBeamSize, parent_states, data);

  EXPECT_TRUE(test_parser->IsReady());
  EXPECT_TRUE(test_parser->IsTerminal());
  EXPECT_EQ(kBeamSize, test_parser->BeamSize());
  EXPECT_EQ(data.size(), test_parser->BatchSize());
  EXPECT_TRUE(test_parser->GetTraceProtos().empty());

  for (int batch_index = 0; batch_index < parent_states.size(); ++batch_index) {
    EXPECT_EQ(0, test_parser->StepsTaken(batch_index));
    const auto &beam = parent_states[batch_index];
    for (int beam_index = 0; beam_index < beam.size(); ++beam_index) {
      // Expect an identity mapping.
      EXPECT_EQ(beam_index,
                test_parser->GetSourceBeamIndex(beam_index, batch_index));
    }
  }

  const auto forwarded_states = test_parser->GetBeam();
  EXPECT_EQ(parent_states, forwarded_states);
}

TEST_F(StatelessComponentTest, UnimplementedMethodsDie) {
  MockTransitionState mock_state_1, mock_state_2, mock_state_3;
  const std::vector<std::vector<const TransitionState *>> parent_states;
  std::vector<string> data;
  for (const string &textproto : {kSentence0, kSentence1, kLongSentence}) {
    Sentence sentence;
    CHECK(TextFormat::ParseFromString(textproto, &sentence));
    data.emplace_back();
    CHECK(sentence.SerializeToString(&data.back()));
  }

  const int kBeamSize = 2;
  auto test_parser = CreateParser(kBeamSize, parent_states, data);

  EXPECT_TRUE(test_parser->IsReady());
  EXPECT_DEATH(test_parser->AdvanceFromPrediction({}, 0, 0),
               "AdvanceFromPrediction not supported");
  EXPECT_DEATH(test_parser->AdvanceFromOracle(),
               "AdvanceFromOracle not supported");
  EXPECT_DEATH(test_parser->GetOracleLabels(), "Method not supported");
  EXPECT_DEATH(test_parser->GetFixedFeatures(nullptr, nullptr, nullptr, 0),
               "Method not supported");
  BulkFeatureExtractor extractor(nullptr, nullptr, nullptr);
  EXPECT_DEATH(test_parser->BulkEmbedFixedFeatures(0, 0, 0, {nullptr}, nullptr),
               "Method not supported");
  EXPECT_DEATH(test_parser->BulkGetFixedFeatures(extractor),
               "Method not supported");
  EXPECT_DEATH(test_parser->GetRawLinkFeatures(0), "Method not supported");
  EXPECT_DEATH(test_parser->AddTranslatedLinkFeaturesToTrace({}, 0),
               "Method not supported");
}

}  // namespace dragnn
}  // namespace syntaxnet
