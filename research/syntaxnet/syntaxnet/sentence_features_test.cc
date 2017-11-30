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

#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/populate_test_inputs.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/utils.h"
#include "syntaxnet/workspace.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

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
  virtual std::vector<string> ExtractMultiFeature(int index) {
    std::vector<string> values;
    FeatureVector result;
    extractor_->ExtractFeatures(workspaces_, sentence_, index,
                                &result);
    values.reserve(result.size());
    for (int i = 0; i < result.size(); ++i) {
      values.push_back(result.type(i)->GetFeatureValueName(result.value(i)));
    }
    return values;
  }

  // Adds an input to the task context.
  void AddInputToContext(const string &name, const string &file_pattern,
                         const string &file_format,
                         const string &record_format) {
    TaskInput *input = context_.GetInput(name);
    TaskInput::Part *part = input->add_part();
    part->set_file_pattern(file_pattern);
    part->set_file_format(file_format);
    part->set_record_format(record_format);
  }

  // Checks that a vector workspace is equal to a target vector.
  void CheckVectorWorkspace(const VectorIntWorkspace &workspace,
                            std::vector<int> target) {
    std::vector<int> src;
    src.reserve(workspace.size());
    for (int i = 0; i < workspace.size(); ++i) {
      src.push_back(workspace.element(i));
    }
    EXPECT_THAT(src, testing::ContainerEq(target));
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
            "  head: 1 label: 'nsubj' break_level: NO_BREAK } "
            "token { word: 'saw' start: 2 end: 4 tag: 'VBD' category: 'VERB'"
            "  label: 'ROOT' break_level: SPACE_BREAK } "
            "token { word: 'a' start: 6 end: 6 tag: 'DT' category: 'DET'"
            "  head: 3 label: 'det' break_level: SPACE_BREAK } "
            "token { word: 'man' start: 8 end: 10 tag: 'NN' category: 'NOUN'"
            "  head: 1 label: 'dobj' break_level: SPACE_BREAK"
            "  [syntaxnet.TokenMorphology.morphology] { "
            "    attribute { name:'morph' value:'Sg' } "
            "    attribute { name:'morph' value:'Masc' } "
            "  } "
            "} "
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

TEST_F(CommonSentenceFeaturesTest, WordFeature) {
  TermFrequencyMap word_map;
  word_map.Increment("saw");
  word_map.Increment("man");
  word_map.Increment("telescope");
  word_map.Increment(".");
  creators_.Add("word-map", "text", "",
                [&](const string &path) { word_map.Save(path); });

  PrepareFeature("word");

  EXPECT_EQ("<OUTSIDE>", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("<UNKNOWN>", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("saw", utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("<UNKNOWN>", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("man", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("<UNKNOWN>", utils::Join(ExtractMultiFeature(4), ","));
  EXPECT_EQ("<UNKNOWN>", utils::Join(ExtractMultiFeature(5), ","));
  EXPECT_EQ("telescope", utils::Join(ExtractMultiFeature(6), ","));
  EXPECT_EQ(".", utils::Join(ExtractMultiFeature(7), ","));
  EXPECT_EQ("<OUTSIDE>", utils::Join(ExtractMultiFeature(8), ","));
}

TEST_F(CommonSentenceFeaturesTest, KnownWordFeature) {
  TermFrequencyMap word_map;
  word_map.Increment("saw");
  word_map.Increment("man");
  word_map.Increment("telescope");
  word_map.Increment(".");
  creators_.Add("known-word-map", "text", "",
                [&](const string &path) { word_map.Save(path); });

  PrepareFeature("known-word");

  // Unlike the "word" feature, does not extract "<OUTSIDE>" or "<UNKNOWN>".
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("saw", utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("man", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(4), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(5), ","));
  EXPECT_EQ("telescope", utils::Join(ExtractMultiFeature(6), ","));
  EXPECT_EQ(".", utils::Join(ExtractMultiFeature(7), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(8), ","));

  EXPECT_EQ(word_map.Size(), extractor_->feature_type(0)->GetDomainSize());
}

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

TEST_F(CommonSentenceFeaturesTest, CharNgramFeature) {
  TermFrequencyMap char_ngram_map;
  for (const string &char_ngram : {"a", "aw", "sa"}) {
    char_ngram_map.Increment(char_ngram);
  }
  creators_.Add(
      "char-ngram-map", "text", "",
      [&char_ngram_map](const string &path) { char_ngram_map.Save(path); });

  PrepareFeature("char-ngram");
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("sa,a,aw", utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("a", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("a", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(8), ","));
}

TEST_F(CommonSentenceFeaturesTest, CharNgramFeatureWithMinLength) {
  TermFrequencyMap char_ngram_map;
  for (const string &char_ngram : {"a", "aw", "sa"}) {
    char_ngram_map.Increment(char_ngram);
  }
  creators_.Add(
      "char-ngram-map", "text", "",
      [&char_ngram_map](const string &path) { char_ngram_map.Save(path); });

  PrepareFeature("char-ngram(min-length=2)");
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("sa,aw", utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(8), ","));
}

TEST_F(CommonSentenceFeaturesTest, CharNgramFeatureWithMaxLength) {
  TermFrequencyMap char_ngram_map;
  for (const string &char_ngram : {"a", "aw", "sa"}) {
    char_ngram_map.Increment(char_ngram);
  }
  creators_.Add(
      "char-ngram-map", "text", "",
      [&char_ngram_map](const string &path) { char_ngram_map.Save(path); });

  PrepareFeature("char-ngram(max-length=1)");
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("a", utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("a", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("a", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(8), ","));
}

TEST_F(CommonSentenceFeaturesTest, CharNgramFeatureWithTerminators) {
  TermFrequencyMap char_ngram_map;
  for (const string &char_ngram :
       {"^", "^s", "^sa", "^saw", "^saw$", "s", "sa", "saw", "a", "^a", "a$",
        "^a$", "aw", "aw$", "w$", "$"}) {
    char_ngram_map.Increment(char_ngram);
  }
  creators_.Add("char-ngram-map", "text", "",
                [&](const string &path) { char_ngram_map.Save(path); });

  PrepareFeature("char-ngram(add-terminators=true)");
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("^,$", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("^,^s,^sa,s,sa,saw,a,aw,aw$,w$,$",
            utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("^,^a,^a$,a,a$,$", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("^,a,$", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(8), ","));
}

TEST_F(CommonSentenceFeaturesTest, CharNgramFeatureWithBoundaries) {
  TermFrequencyMap char_ngram_map;
  for (const string &char_ngram :
       {"^ ", "^ s", "^ sa", "^ saw", "^ saw $", "s", "sa", "saw", "a", "^ a",
        "a $", "^ a $", "aw", "aw $", "w $", " $"}) {
    char_ngram_map.Increment(char_ngram);
  }
  creators_.Add("char-ngram-map", "text", "",
                [&](const string &path) { char_ngram_map.Save(path); });

  PrepareFeature("char-ngram(mark-boundaries=true)");
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("^ s,^ sa,^ saw $,a,aw $,w $",
            utils::Join(ExtractMultiFeature(1), ","));
  EXPECT_EQ("^ a $", utils::Join(ExtractMultiFeature(2), ","));
  EXPECT_EQ("a", utils::Join(ExtractMultiFeature(3), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(8), ","));
}

TEST_F(CommonSentenceFeaturesTest, MorphologySetFeature) {
  TermFrequencyMap morphology_map;
  morphology_map.Increment("morph=Sg");
  morphology_map.Increment("morph=Sg");
  morphology_map.Increment("morph=Masc");
  morphology_map.Increment("morph=Masc");
  morphology_map.Increment("morph=Pl");
  creators_.Add(
      "morphology-map", "text", "",
      [&morphology_map](const string &path) { morphology_map.Save(path); });

  PrepareFeature("morphology-set");
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(-1), ","));
  EXPECT_EQ("", utils::Join(ExtractMultiFeature(0), ","));
  EXPECT_EQ("morph=Sg,morph=Masc", utils::Join(ExtractMultiFeature(3), ","));
}

TEST_F(CommonSentenceFeaturesTest, CapitalizationProcessesCorrectly) {
  Capitalization feature;
  feature.RequestWorkspaces(&registry_);
  workspaces_.Reset(registry_);
  feature.Preprocess(&workspaces_, &sentence_);

  // Check the workspace contains what we expect.
  EXPECT_TRUE(workspaces_.Has<VectorIntWorkspace>(feature.Workspace()));
  const VectorIntWorkspace &workspace =
      workspaces_.Get<VectorIntWorkspace>(feature.Workspace());
  constexpr int UPPERCASE = Capitalization::UPPERCASE;
  constexpr int LOWERCASE = Capitalization::LOWERCASE;
  constexpr int NON_ALPHABETIC = Capitalization::NON_ALPHABETIC;
  CheckVectorWorkspace(workspace,
                       {UPPERCASE, LOWERCASE, LOWERCASE, LOWERCASE, LOWERCASE,
                        LOWERCASE, LOWERCASE, NON_ALPHABETIC});
}

class CharFeatureTest : public SentenceFeaturesTest {
 protected:
  CharFeatureTest()
      : SentenceFeaturesTest(
          "text: '一 个 测 试 员  ' "
          "token { word: '一' start: 0 end: 2 } "
          "token { word: '个' start: 3 end: 5 } "
          "token { word: '测' start: 6 end: 8 } "
          "token { word: '试' start: 9 end: 11 } "
          "token { word: '员' start: 12 end: 14 } "
          "token { word: ' ' start: 15 end: 15 } "
          "token { word: '\t' start: 16 end: 16 } ") {}
};

TEST_F(CharFeatureTest, CharFeature) {
  TermFrequencyMap char_map;
  char_map.Increment("一");
  char_map.Increment("个");
  char_map.Increment("试");
  char_map.Increment("员");
  creators_.Add(
      "char-map", "text", "",
      [&char_map](const string &path) { char_map.Save(path); });

  // Test that Char works as expected.
  PrepareFeature("char");
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(-1));
  EXPECT_EQ("一", ExtractFeature(0));
  EXPECT_EQ("个", ExtractFeature(1));
  EXPECT_EQ("<UNKNOWN>", ExtractFeature(2));  // "测" is not in the char map.
  EXPECT_EQ("试", ExtractFeature(3));
  EXPECT_EQ("员", ExtractFeature(4));
  EXPECT_EQ("<BREAK_CHAR>", ExtractFeature(5));
  EXPECT_EQ("<BREAK_CHAR>", ExtractFeature(6));
  EXPECT_EQ("<OUTSIDE>", ExtractFeature(7));
}

}  // namespace syntaxnet
