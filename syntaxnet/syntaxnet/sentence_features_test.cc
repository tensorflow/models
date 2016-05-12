/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/sentence_features.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "syntaxnet/utils.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/populate_test_inputs.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/workspace.h"

using testing::UnorderedElementsAreArray;

namespace syntaxnet {

// A basic fixture for testing Features. Takes a string of a
// Sentence protobuf that is used as the test data in the constructor.
class SentenceFeaturesTest : public ::testing::Test {
 protected:
  explicit SentenceFeaturesTest(const string &prototxt)
      : sentence_(ParseASCII(prototxt)),
        creators_(PopulateTestInputs::Defaults(sentence_)) {}

  static Sentence ParseASCII(const string &prototxt) {
    Sentence document;
    CHECK(TextFormat::ParseFromString(prototxt, &document));
    return document;
  }

  // Prepares a new feature for extracting from the attached sentence,
  // regenerating the TaskContext and all resources. Will automatically add
  // anything in info_ field into the LexiFuse repository.
  virtual void PrepareFeature(const string &fml) {
    context_.mutable_spec()->mutable_input()->Clear();
    context_.mutable_spec()->mutable_output()->Clear();
    extractor_.reset(new SentenceExtractor());
    extractor_->Parse(fml);
    extractor_->Setup(&context_);
    creators_.Populate(&context_);
    extractor_->Init(&context_);
    extractor_->RequestWorkspaces(&registry_);
    workspaces_.Reset(registry_);
    extractor_->Preprocess(&workspaces_, &sentence_);
  }

  // Returns the string representation of the prepared feature extracted at the
  // given index.
  virtual string ExtractFeature(int index) {
    FeatureVector result;
    extractor_->ExtractFeatures(workspaces_, sentence_, index,
                                &result);
    return result.type(0)->GetFeatureValueName(result.value(0));
  }

  // Extracts a vector of string representations from evaluating the prepared
  // set feature (returning multiple values) at the given index.
  virtual vector<string> ExtractMultiFeature(int index) {
    vector<string> values;
    FeatureVector result;
    extractor_->ExtractFeatures(workspaces_, sentence_, index,
                                &result);
    for (int i = 0; i < result.size(); ++i) {
      values.push_back(result.type(i)->GetFeatureValueName(result.value(i)));
    }
    return values;
  }

  Sentence sentence_;
  WorkspaceSet workspaces_;

  PopulateTestInputs::CreatorMap creators_;
  TaskContext context_;
  WorkspaceRegistry registry_;
  std::unique_ptr<SentenceExtractor> extractor_;
};

// Test fixture for simple common features that operate on just a sentence.
class CommonSentenceFeaturesTest : public SentenceFeaturesTest {
 protected:
  CommonSentenceFeaturesTest()
      : SentenceFeaturesTest(
            "text: 'I saw a man with a telescope.' "
            "token { word: 'I' start: 0 end: 0 tag: 'PRP' category: 'PRON'"
            " head: 1 label: 'nsubj' break_level: NO_BREAK } "
            "token { word: 'saw' start: 2 end: 4 tag: 'VBD' category: 'VERB'"
            " label: 'ROOT' break_level: SPACE_BREAK } "
            "token { word: 'a' start: 6 end: 6 tag: 'DT' category: 'DET'"
            " head: 3 label: 'det' break_level: SPACE_BREAK } "
            "token { word: 'man' start: 8 end: 10 tag: 'NN' category: 'NOUN'"
            " head: 1 label: 'dobj' break_level: SPACE_BREAK } "
            "token { word: 'with' start: 12 end: 15 tag: 'IN' category: 'ADP'"
            " head: 1 label: 'prep' break_level: SPACE_BREAK } "
            "token { word: 'a' start: 17 end: 17 tag: 'DT' category: 'DET'"
            " head: 6 label: 'det' break_level: SPACE_BREAK } "
            "token { word: 'telescope' start: 19 end: 27 tag: 'NN' category: "
            "'NOUN'"
            " head: 4 label: 'pobj'  break_level: SPACE_BREAK } "
            "token { word: '.' start: 28 end: 28 tag: '.' category: '.'"
            " head: 1 label: 'p' break_level: NO_BREAK }") {}
};

TEST_F(CommonSentenceFeaturesTest, TagFeature) {
  PrepareFeature("tag");
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(-1));
  EXPECT_EQ("PRP", ExtractFeature(0));
  EXPECT_EQ("VBD", ExtractFeature(1));
  EXPECT_EQ("DT", ExtractFeature(2));
  EXPECT_EQ("NN", ExtractFeature(3));
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(8));
}

TEST_F(CommonSentenceFeaturesTest, TagFeaturePassesArgs) {
  PrepareFeature("tag(min-freq=5)");  // don't load any tags
  EXPECT_EQ(ExtractFeature(-1), "<OUTSIDE>");
  EXPECT_EQ(ExtractFeature(0), "<UNKNOWN>");
  EXPECT_EQ(ExtractFeature(8), "<OUTSIDE>");

  // Only 2 features: <UNKNOWN> and <OUTSIDE>.
  EXPECT_EQ(2, extractor_->feature_type(0)->GetDomainSize());
}

TEST_F(CommonSentenceFeaturesTest, OffsetPlusTag) {
  PrepareFeature("offset(-1).tag(min-freq=2)");
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(-1));
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(0));
  EXPECT_EQ("<UNKNOWN>", ExtractFeature(1));
  EXPECT_EQ("<UNKNOWN>", ExtractFeature(2));
  EXPECT_EQ("DT", ExtractFeature(3));  // DT, NN are the only freq tags
  EXPECT_EQ("NN", ExtractFeature(4));
  EXPECT_EQ("<UNKNOWN>", ExtractFeature(5));
  EXPECT_EQ("DT", ExtractFeature(6));
  EXPECT_EQ("NN", ExtractFeature(7));
  EXPECT_EQ("<UNKNOWN>", ExtractFeature(8));
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(9));
}

}  // namespace syntaxnet
