#include "neurosis/parser_features.h"

#include <string>

#include "neurosis/utils.h"
#include "neurosis/feature_extractor.h"
#include "neurosis/parser_state.h"
#include "neurosis/populate_test_inputs.h"
#include "neurosis/sentence.pb.h"
#include "task_context.h"
#include "task_spec.pb.h"
#include "term_frequency_map.h"
#include "neurosis/workspace.h"
#include "testing/base/public/googletest.h"
#include "testing/base/public/gunit.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace neurosis {

// Feature extractor for the transition parser based on a parser state object.
typedef FeatureExtractor<ParserState, int> ParserIndexFeatureExtractor;

// Test fixture for parser features.
class ParserFeatureFunctionTest : public ::testing::Test {
 protected:
  // Sets up a parser state.
  void SetUp() override {
    // Prepare a document.
    const char *kTaggedDocument =
        "text: 'I saw a man with a telescope.' "
        "token { word: 'I' start: 0 end: 0 tag: 'PRP' category: 'PRON'"
        " break_level: NO_BREAK } "
        "token { word: 'saw' start: 2 end: 4 tag: 'VBD' category: 'VERB'"
        " break_level: SPACE_BREAK } "
        "token { word: 'a' start: 6 end: 6 tag: 'DT' category: 'DET'"
        " break_level: SPACE_BREAK } "
        "token { word: 'man' start: 8 end: 10 tag: 'NN' category: 'NOUN'"
        " break_level: SPACE_BREAK } "
        "token { word: 'with' start: 12 end: 15 tag: 'IN' category: 'ADP'"
        " break_level: SPACE_BREAK } "
        "token { word: 'a' start: 17 end: 17 tag: 'DT' category: 'DET'"
        " break_level: SPACE_BREAK } "
        "token { word: 'telescope' start: 19 end: 27 tag: 'NN' category: 'NOUN'"
        " break_level: SPACE_BREAK } "
        "token { word: '.' start: 28 end: 28 tag: '.' category: '.'"
        " break_level: NO_BREAK }";
    CHECK(sentence_.ParseASCII(kTaggedDocument));
    creators_ = PopulateTestInputs::Defaults(sentence_);

    // Prepare a label map. By adding labels in lexicographic order we make sure
    // the term indices stay the same after sorting (which happens when the
    // label map is saved to disk).
    label_map_.Increment("NULL");
    label_map_.Increment("ROOT");
    label_map_.Increment("det");
    label_map_.Increment("dobj");
    label_map_.Increment("nsubj");
    label_map_.Increment("p");
    label_map_.Increment("pobj");
    label_map_.Increment("prep");
    creators_.Add("label-map", "text", "", [this](const string &filename) {
      label_map_.Save(filename);
    });

    // Prepare a parser state.
    state_.reset(new ParserState(&sentence_, nullptr /* no transition state */,
                                 &label_map_));
  }

  // Prepares a feature for computations.
  string ExtractFeature(const string &feature_name) {
    context_.mutable_spec()->mutable_input()->Clear();
    context_.mutable_spec()->mutable_output()->Clear();
    feature_extractor_.reset(new ParserFeatureExtractor());
    feature_extractor_->Parse(feature_name);
    feature_extractor_->Setup(&context_);
    creators_.Populate(&context_);
    feature_extractor_->Init(&context_);
    feature_extractor_->RequestWorkspaces(&registry_);
    workspaces_.Reset(registry_);
    feature_extractor_->Preprocess(&workspaces_, state_.get());
    FeatureVector result;
    feature_extractor_->ExtractFeatures(workspaces_, *state_, &result);
    return result.type(0)->GetFeatureValueName(result.value(0));
  }

  std::unique_ptr<ParserState> state_;
  Sentence sentence_;
  WorkspaceSet workspaces_;
  TermFrequencyMap label_map_;

  PopulateTestInputs::CreatorMap creators_;
  TaskContext context_;
  WorkspaceRegistry registry_;
  std::unique_ptr<ParserFeatureExtractor> feature_extractor_;
};

TEST_F(ParserFeatureFunctionTest, TagFeatureFunction) {
  state_->Push(-1);
  state_->Push(0);
  EXPECT_EQ("PRP", ExtractFeature("input.tag"));
  EXPECT_EQ("VBD", ExtractFeature("input(1).tag"));
  EXPECT_EQ("<OUTSIDE>", ExtractFeature("input(10).tag"));
  EXPECT_EQ("PRP", ExtractFeature("stack(0).tag"));
  EXPECT_EQ("<ROOT>", ExtractFeature("stack(1).tag"));
}

TEST_F(ParserFeatureFunctionTest, LabelFeatureFunction) {
  // Construct a partial dependency tree.
  state_->AddArc(0, 1, 4);
  state_->AddArc(1, -1, 1);
  state_->AddArc(2, 3, 2);
  state_->AddArc(3, 1, 3);
  state_->AddArc(5, 6, 2);
  state_->AddArc(6, 4, 6);
  state_->AddArc(7, 1, 5);

  // Test the feature function.
  EXPECT_EQ(label_map_.GetTerm(4), ExtractFeature("input.label"));
  EXPECT_EQ("ROOT", ExtractFeature("input(1).label"));
  EXPECT_EQ(label_map_.GetTerm(2), ExtractFeature("input(2).label"));

  // Push artifical root token onto the stack. This triggers the wrapped <ROOT>
  // value, rather than indicating a token with the label "ROOT" (which may or
  // may not be the artificial root token.)
  state_->Push(-1);
  EXPECT_EQ("<ROOT>", ExtractFeature("stack.label"));
}

}  // namespace neurosis
