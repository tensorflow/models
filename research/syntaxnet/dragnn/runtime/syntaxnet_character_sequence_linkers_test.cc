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
#include "dragnn/runtime/sequence_linker.h"
#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::ElementsAre;

// Returns a ComponentSpec parsed from the |text|.
ComponentSpec ParseSpec(const string &text) {
  ComponentSpec component_spec;
  CHECK(TextFormat::ParseFromString(text, &component_spec));
  return component_spec;
}

// Returns a ComponentSpec that some linker supports.
ComponentSpec MakeSupportedSpec() {
  return ParseSpec(R"(
    transition_system { registered_name:'shift-only' }
    backend { registered_name:'SyntaxNetComponent' }
    linked_feature { fml:'input.first-char-focus' source_translator:'identity' }
  )");
}

// Returns a Sentence parsed from the |text|.
Sentence ParseSentence(const string &text) {
  Sentence sentence;
  CHECK(TextFormat::ParseFromString(text, &sentence));
  return sentence;
}

// Returns a default sentence.
Sentence MakeSentence() {
  return ParseSentence(R"(
    text:'012345678901234567890123456789人1工神2经网¢络'
    token { start:30 end:36 word:'人1工' }
    token { start:37 end:43 word:'神2经' }
    token { start:44 end:51 word:'网¢络' }
  )");
}

// Number of UTF-8 characters in the default sentence.
constexpr int kNumChars = 9;

// Tests that the linker supports appropriate specs.
TEST(SyntaxNetCharacterSequenceLinkersTest, Supported) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetFirstCharacterIdentitySequenceLinker");

  channel.set_source_translator("reverse-char");
  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetFirstCharacterReversedSequenceLinker");

  channel.set_fml("input.last-char-focus");
  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetLastCharacterReversedSequenceLinker");

  channel.set_source_translator("identity");
  TF_ASSERT_OK(SequenceLinker::Select(channel, component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetLastCharacterIdentitySequenceLinker");
}

// Tests that the linker requires the right transition system.
TEST(SyntaxNetCharacterSequenceLinkersTest, WrongTransitionSystem) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  const LinkedFeatureChannel &channel = component_spec.linked_feature(0);

  component_spec.mutable_backend()->set_registered_name("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker requires the right FML.
TEST(SyntaxNetCharacterSequenceLinkersTest, WrongFml) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  channel.set_fml("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker requires the right translator.
TEST(SyntaxNetCharacterSequenceLinkersTest, WrongTranslator) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  LinkedFeatureChannel &channel = *component_spec.mutable_linked_feature(0);

  channel.set_source_translator("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Tests that the linker requires the right backend.
TEST(SyntaxNetCharacterSequenceLinkersTest, WrongBackend) {
  string name;
  ComponentSpec component_spec = MakeSupportedSpec();
  const LinkedFeatureChannel &channel = component_spec.linked_feature(0);

  component_spec.mutable_backend()->set_registered_name("bad");
  EXPECT_THAT(SequenceLinker::Select(channel, component_spec, &name),
              test::IsErrorWithSubstr("No SequenceLinker supports channel"));
}

// Rig for testing GetLinks().
class SyntaxNetCharacterSequenceLinkersGetLinksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize() doesn't look at the channel or spec, so use empty protos.
    const ComponentSpec component_spec;
    const LinkedFeatureChannel channel;
    TF_ASSERT_OK(
        SequenceLinker::New("SyntaxNetFirstCharacterIdentitySequenceLinker",
                            channel, component_spec, &first_identity_));
    TF_ASSERT_OK(
        SequenceLinker::New("SyntaxNetFirstCharacterReversedSequenceLinker",
                            channel, component_spec, &first_reversed_));
    TF_ASSERT_OK(
        SequenceLinker::New("SyntaxNetLastCharacterIdentitySequenceLinker",
                            channel, component_spec, &last_identity_));
    TF_ASSERT_OK(
        SequenceLinker::New("SyntaxNetLastCharacterReversedSequenceLinker",
                            channel, component_spec, &last_reversed_));
  }

  // Linkers in all four configurations.
  std::unique_ptr<SequenceLinker> first_identity_;
  std::unique_ptr<SequenceLinker> first_reversed_;
  std::unique_ptr<SequenceLinker> last_identity_;
  std::unique_ptr<SequenceLinker> last_reversed_;
};

// Tests that the linkers can extract links from the default sentence.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest, DefaultSentence) {
  const Sentence sentence = MakeSentence();
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> links = {123, 456, 789};  // gets overwritten

  TF_ASSERT_OK(first_identity_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(0, 3, 6));
  TF_ASSERT_OK(first_reversed_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(8, 5, 2));
  TF_ASSERT_OK(last_identity_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(2, 5, 8));
  TF_ASSERT_OK(last_reversed_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(6, 3, 0));
}

// Tests that the linkers can handle an empty sentence.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest, EmptySentence) {
  const Sentence sentence;
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> links;

  TF_ASSERT_OK(first_identity_->GetLinks(kNumChars, &input, &links));
  TF_ASSERT_OK(first_reversed_->GetLinks(kNumChars, &input, &links));
  TF_ASSERT_OK(last_identity_->GetLinks(kNumChars, &input, &links));
  TF_ASSERT_OK(last_reversed_->GetLinks(kNumChars, &input, &links));
}

// Tests that the linkers fail if the batch is not a singleton.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest, NonSingleton) {
  const Sentence sentence = MakeSentence();
  const std::vector<string> data = {sentence.SerializeAsString(),
                                    sentence.SerializeAsString()};
  InputBatchCache input(data);
  std::vector<int32> links;

  EXPECT_THAT(first_identity_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr("Non-singleton batch: got 2 elements"));
  EXPECT_THAT(first_reversed_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr("Non-singleton batch: got 2 elements"));
  EXPECT_THAT(last_identity_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr("Non-singleton batch: got 2 elements"));
  EXPECT_THAT(last_reversed_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr("Non-singleton batch: got 2 elements"));
}

// Tests that the linkers fail if the first token starts in the middle of a
// UTF-8 character.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest, FirstTokenStartsWrong) {
  Sentence sentence = MakeSentence();
  sentence.mutable_token(0)->set_start(31);
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> links;

  EXPECT_THAT(first_identity_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "First token starts in the middle of a UTF-8 character"));
  EXPECT_THAT(first_reversed_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "First token starts in the middle of a UTF-8 character"));
  EXPECT_THAT(last_identity_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "First token starts in the middle of a UTF-8 character"));
  EXPECT_THAT(last_reversed_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "First token starts in the middle of a UTF-8 character"));
}

// Tests that the linkers fail if the last token ends in the middle of a UTF-8
// character.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest, LastTokenEndsWrong) {
  Sentence sentence = MakeSentence();
  sentence.mutable_token(2)->set_end(45);
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> links;

  EXPECT_THAT(first_identity_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "Last token ends in the middle of a UTF-8 character"));
  EXPECT_THAT(first_reversed_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "Last token ends in the middle of a UTF-8 character"));
  EXPECT_THAT(last_identity_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "Last token ends in the middle of a UTF-8 character"));
  EXPECT_THAT(last_reversed_->GetLinks(kNumChars, &input, &links),
              test::IsErrorWithSubstr(
                  "Last token ends in the middle of a UTF-8 character"));
}

// Tests that the linkers can tolerate a sentence where the interior token byte
// offsets are wrong.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest,
       InteriorTokenBoundariesSlightlyWrong) {
  Sentence sentence = MakeSentence();
  sentence.mutable_token(0)->set_end(35);
  sentence.mutable_token(1)->set_start(38);
  sentence.mutable_token(1)->set_end(42);
  sentence.mutable_token(2)->set_start(45);
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> links;

  // The results should be the same as in the default sentence.
  TF_ASSERT_OK(first_identity_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(0, 3, 6));
  TF_ASSERT_OK(first_reversed_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(8, 5, 2));
  TF_ASSERT_OK(last_identity_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(2, 5, 8));
  TF_ASSERT_OK(last_reversed_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(6, 3, 0));
}

// As above, but places the token boundaries even further off.
TEST_F(SyntaxNetCharacterSequenceLinkersGetLinksTest,
       InteriorTokenBoundariesMostlyWrong) {
  Sentence sentence = MakeSentence();
  sentence.mutable_token(0)->set_end(34);
  sentence.mutable_token(1)->set_start(39);
  sentence.mutable_token(1)->set_end(41);
  sentence.mutable_token(2)->set_start(46);
  InputBatchCache input(sentence.SerializeAsString());
  std::vector<int32> links;

  // The results should be the same as in the default sentence.
  TF_ASSERT_OK(first_identity_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(0, 3, 6));
  TF_ASSERT_OK(first_reversed_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(8, 5, 2));
  TF_ASSERT_OK(last_identity_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(2, 5, 8));
  TF_ASSERT_OK(last_reversed_->GetLinks(kNumChars, &input, &links));
  EXPECT_THAT(links, ElementsAre(6, 3, 0));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
