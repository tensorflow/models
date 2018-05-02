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

#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/test/term_map_helpers.h"
#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr char kResourceName[] = "word-map";

// Returns a ComponentSpec parsed from the |text| that contains a term map
// resource pointing at the |path|.
ComponentSpec MakeSpec(const string &text, const string &path) {
  ComponentSpec component_spec;
  CHECK(TextFormat::ParseFromString(text, &component_spec));
  AddTermMapResource(kResourceName, path, &component_spec);
  return component_spec;
}

// Returns a ComponentSpec that the extractor will support.
ComponentSpec MakeSupportedSpec() {
  return MakeSpec(
      R"(transition_system { registered_name: 'shift-only' }
         backend { registered_name: 'SyntaxNetComponent' }
         fixed_feature {}  # breaks hard-coded refs to channel 0
         fixed_feature { size: 1 fml: 'input.token.word(min-freq=2)' })",
      "/dev/null");
}

// Returns a default sentence.
Sentence MakeSentence() {
  Sentence sentence;
  for (const string &word : {"a", "bc", "def"}) {
    Token *token = sentence.add_token();
    token->set_start(0);  // never used; set because required field
    token->set_end(0);  // never used; set because required field
    token->set_word(word);
  }
  return sentence;
}

// Tests that the extractor supports an appropriate spec.
TEST(SyntaxNetWordSequenceExtractorTest, Supported) {
  const ComponentSpec component_spec = MakeSupportedSpec();
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  string name;
  TF_ASSERT_OK(SequenceExtractor::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetWordSequenceExtractor");
}

// Tests that the extractor requires the proper backend.
TEST(SyntaxNetWordSequenceExtractorTest, WrongBackend) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_backend()->set_registered_name("bad");
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  string name;
  EXPECT_THAT(
      SequenceExtractor::Select(channel, component_spec, &name),
      test::IsErrorWithSubstr("No SequenceExtractor supports channel"));
}

// Tests that the extractor requires the proper transition system.
TEST(SyntaxNetWordSequenceExtractorTest, WrongTransitionSystem) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_transition_system()->set_registered_name("bad");
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  string name;
  EXPECT_THAT(
      SequenceExtractor::Select(channel, component_spec, &name),
      test::IsErrorWithSubstr("No SequenceExtractor supports channel"));
}

// Expects that the |fml| is rejected by the extractor.
void ExpectRejectedFml(const string &fml) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_fixed_feature(1)->set_fml(fml);
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  string name;
  EXPECT_THAT(
      SequenceExtractor::Select(channel, component_spec, &name),
      test::IsErrorWithSubstr("No SequenceExtractor supports channel"));
}

// Tests that the extractor requires the proper FML.
TEST(SyntaxNetWordSequenceExtractorTest, WrongFml) {
  ExpectRejectedFml("bad");
  EXPECT_DEATH(ExpectRejectedFml("input.token.word("),
               "Error in feature model");
  EXPECT_DEATH(ExpectRejectedFml("input.token.word()"),
               "Error in feature model");
  ExpectRejectedFml("input.token.word(10)");
  EXPECT_DEATH(ExpectRejectedFml("input.token.word(min-freq=)"),
               "Error in feature model");
  EXPECT_DEATH(ExpectRejectedFml("input.token.word(min-freq=10"),
               "Error in feature model");
  ExpectRejectedFml("input.token.word(min-freq=ten)");
  ExpectRejectedFml("input.token.word(min_freq=10)");  // underscore
}

// Tests that the extractor can be initialized and used to extract feature IDs.
TEST(SyntaxNetWordSequenceExtractorTest, InitializeAndGetIds) {
  // Terms are sorted by descending frequency, so this ensures a=0, bc=1, etc.
  // Note that "e" is too infrequent, so vocabulary_size=5 from 3 terms plus 2
  // special values.
  const string path = WriteTermMap({{"a", 5}, {"bc", 3}, {"d", 2}, {"e", 1}});
  const ComponentSpec component_spec = MakeSpec(
      "fixed_feature {} "
      "fixed_feature { vocabulary_size:5 fml:'input.token.word(min-freq=2)' }",
      path);
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  std::unique_ptr<SequenceExtractor> extractor;
  TF_ASSERT_OK(SequenceExtractor::New("SyntaxNetWordSequenceExtractor", channel,
                                      component_spec, &extractor));

  const Sentence sentence = MakeSentence();
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> ids;
  TF_ASSERT_OK(extractor->GetIds(&input, &ids));

  const std::vector<int32> expected_ids = {0, 1, 3};
  EXPECT_EQ(ids, expected_ids);
}

// Tests that an empty term map works.
TEST(SyntaxNetWordSequenceExtractorTest, EmptyTermMap) {
  const string path = WriteTermMap({});
  const ComponentSpec component_spec = MakeSpec(
      "fixed_feature {} "
      "fixed_feature { fml:'input.token.word' vocabulary_size:2 }",
      path);
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  std::unique_ptr<SequenceExtractor> extractor;
  TF_ASSERT_OK(SequenceExtractor::New("SyntaxNetWordSequenceExtractor", channel,
                                      component_spec, &extractor));

  const Sentence sentence = MakeSentence();
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> ids = {1, 2, 3, 4};  // should be overwritten
  TF_ASSERT_OK(extractor->GetIds(&input, &ids));

  const std::vector<int32> expected_ids = {0, 0, 0};
  EXPECT_EQ(ids, expected_ids);
}

// Tests that GetIds() fails if the batch is the wrong size.
TEST(SyntaxNetWordSequenceExtractorTest, WrongBatchSize) {
  const string path = WriteTermMap({});
  const ComponentSpec component_spec = MakeSpec(
      "fixed_feature {} "
      "fixed_feature { fml:'input.token.word' vocabulary_size:2 }",
      path);
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  std::unique_ptr<SequenceExtractor> extractor;
  TF_ASSERT_OK(SequenceExtractor::New("SyntaxNetWordSequenceExtractor", channel,
                                      component_spec, &extractor));

  const Sentence sentence = MakeSentence();
  const std::vector<string> data = {sentence.SerializeAsString(),
                                    sentence.SerializeAsString()};
  InputBatchCache input(data);
  std::vector<int32> ids;
  EXPECT_THAT(extractor->GetIds(&input, &ids),
              test::IsErrorWithSubstr("Non-singleton batch: got 2 elements"));
}

// Tests that initialization fails if the vocabulary size does not match.
TEST(SyntaxNetWordSequenceExtractorTest, WrongVocabularySize) {
  const string path = WriteTermMap({});
  const ComponentSpec component_spec = MakeSpec(
      "fixed_feature {} "
      "fixed_feature { fml:'input.token.word' vocabulary_size:1000 }",
      path);
  const FixedFeatureChannel &channel = component_spec.fixed_feature(1);

  std::unique_ptr<SequenceExtractor> extractor;
  EXPECT_THAT(
      SequenceExtractor::New("SyntaxNetWordSequenceExtractor", channel,
                             component_spec, &extractor),
      test::IsErrorWithSubstr("Word vocabulary size mismatch between term "
                              "map (2) and ComponentSpec (1000)"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
