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

// Features that operate on Sentence objects. Most features are defined
// in this header so they may be re-used via composition into other more
// advanced feature classes.

#ifndef $TARGETDIR_SENTENCE_FEATURES_H_
#define $TARGETDIR_SENTENCE_FEATURES_H_

#include "syntaxnet/affix.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/feature_types.h"
#include "syntaxnet/shared_store.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/workspace.h"

namespace syntaxnet {

// Feature function for any component that processes Sentences, whose
// focus is a token index into the sentence.
typedef FeatureFunction<Sentence, int> SentenceFeature;

// Alias for Locator type features that take (Sentence, int) signatures
// and call other (Sentence, int) features.
template <class DER>
using Locator = FeatureLocator<DER, Sentence, int>;

class TokenLookupFeature : public SentenceFeature {
 public:
  void Init(TaskContext *context) override {
    set_feature_type(new ResourceBasedFeatureType<TokenLookupFeature>(
        name(), this, {{NumValues(), "<OUTSIDE>"}}));
  }

  // Given a position in a sentence and workspaces, looks up the corresponding
  // feature value. The index is relative to the start of the sentence.
  virtual FeatureValue ComputeValue(const Token &token) const = 0;

  // Number of unique values.
  virtual int64 NumValues() const = 0;

  // Convert the numeric value of the feature to a human readable string.
  virtual string GetFeatureValueName(FeatureValue value) const = 0;

  // Name of the shared workspace.
  virtual string WorkspaceName() const = 0;

  // Runs ComputeValue for each token in the sentence.
  void Preprocess(WorkspaceSet *workspaces,
                  Sentence *sentence) const override {
    if (workspaces->Has<VectorIntWorkspace>(workspace_)) return;
    VectorIntWorkspace *workspace = new VectorIntWorkspace(
        sentence->token_size());
    for (int i = 0; i < sentence->token_size(); ++i) {
      const int value = ComputeValue(sentence->token(i));
      workspace->set_element(i, value);
    }
    workspaces->Set<VectorIntWorkspace>(workspace_, workspace);
  }

  // Requests a vector of int's to store in the workspace registry.
  void RequestWorkspaces(WorkspaceRegistry *registry) override {
    workspace_ = registry->Request<VectorIntWorkspace>(WorkspaceName());
  }

  // Returns the precomputed value, or NumValues() for features outside
  // the sentence.
  FeatureValue Compute(const WorkspaceSet &workspaces,
                       const Sentence &sentence, int focus,
                       const FeatureVector *result) const override {
    if (focus < 0 || focus >= sentence.token_size()) return NumValues();
    return workspaces.Get<VectorIntWorkspace>(workspace_).element(focus);
  }

 private:
  int workspace_;
};

// Lookup feature that uses a TermFrequencyMap to store a string->int mapping.
class TermFrequencyMapFeature : public TokenLookupFeature {
 public:
  explicit TermFrequencyMapFeature(const string &input_name)
      : input_name_(input_name), min_freq_(0), max_num_terms_(0) {}
  ~TermFrequencyMapFeature() override;

  // Requests the input map as a resource.
  void Setup(TaskContext *context) override;

  // Loads the input map into memory (using SharedStore to avoid redundancy.)
  void Init(TaskContext *context) override;

  // Number of unique values.
  virtual int64 NumValues() const { return term_map_->Size() + 1; }

  // Special value for strings not in the map.
  FeatureValue UnknownValue() const { return term_map_->Size(); }

  // Uses the TermFrequencyMap to lookup the string associated with a value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Name of the shared workspace.
  string WorkspaceName() const override;

 protected:
  const TermFrequencyMap &term_map() const { return *term_map_; }

 private:
  // Shortcut pointer to shared map. Not owned.
  const TermFrequencyMap *term_map_ = nullptr;

  // Name of the input for the term map.
  string input_name_;

  // Filename of the underlying resource.
  string file_name_;

  // Minimum frequency for term map.
  int min_freq_;

  // Maximum number of terms for term map.
  int max_num_terms_;
};

class Word : public TermFrequencyMapFeature {
 public:
  Word() : TermFrequencyMapFeature("word-map") {}

  FeatureValue ComputeValue(const Token &token) const override {
    string form = token.word();
    return term_map().LookupIndex(form, UnknownValue());
  }
};

class LowercaseWord : public TermFrequencyMapFeature {
 public:
  LowercaseWord() : TermFrequencyMapFeature("lc-word-map") {}

  FeatureValue ComputeValue(const Token &token) const override {
    const string lcword = utils::Lowercase(token.word());
    return term_map().LookupIndex(lcword, UnknownValue());
  }
};

class Tag : public TermFrequencyMapFeature {
 public:
  Tag() : TermFrequencyMapFeature("tag-map") {}

  FeatureValue ComputeValue(const Token &token) const override {
    return term_map().LookupIndex(token.tag(), UnknownValue());
  }
};

class Label : public TermFrequencyMapFeature {
 public:
  Label() : TermFrequencyMapFeature("label-map") {}

  FeatureValue ComputeValue(const Token &token) const override {
    return term_map().LookupIndex(token.label(), UnknownValue());
  }
};

class LexicalCategoryFeature : public TokenLookupFeature {
 public:
  LexicalCategoryFeature(const string &name, int cardinality)
      : name_(name), cardinality_(cardinality) {}
  ~LexicalCategoryFeature() override {}

  FeatureValue NumValues() const override { return cardinality_; }

  // Returns the identifier for the workspace for this preprocessor.
  string WorkspaceName() const override {
    return tensorflow::strings::StrCat(name_, ":", cardinality_);
  }

 private:
  // Name of the category type.
  const string name_;

  // Number of values.
  const int cardinality_;
};

// Preprocessor that computes whether a word has a hyphen or not.
class Hyphen : public LexicalCategoryFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_HYPHEN = 0,
    HAS_HYPHEN = 1,
    CARDINALITY = 2,
  };

  // Default constructor.
  Hyphen() : LexicalCategoryFeature("hyphen", CARDINALITY) {}

  // Returns a string representation of the enum value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the category value for the token.
  FeatureValue ComputeValue(const Token &token) const override;
};

// Preprocessor that computes whether a word has a hyphen or not.
class Digit : public LexicalCategoryFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_DIGIT = 0,
    SOME_DIGIT = 1,
    ALL_DIGIT = 2,
    CARDINALITY = 3,
  };

  // Default constructor.
  Digit() : LexicalCategoryFeature("digit", CARDINALITY) {}

  // Returns a string representation of the enum value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the category value for the token.
  FeatureValue ComputeValue(const Token &token) const override;
};

// TokenLookupPreprocessor object to compute prefixes and suffixes of words. The
// AffixTable is stored in the SharedStore. This is very similar to the
// implementation of TermFrequencyMapPreprocessor, but using an AffixTable to
// perform the lookups. There are only two specializations, for prefixes and
// suffixes.
class AffixTableFeature : public TokenLookupFeature {
 public:
  // Explicit constructor to set the type of the table. This determines the
  // requested input.
  explicit AffixTableFeature(AffixTable::Type type);
  ~AffixTableFeature() override;

  // Requests inputs for the affix table.
  void Setup(TaskContext *context) override;

  // Loads the affix table from the SharedStore.
  void Init(TaskContext *context) override;

  // The workspace name is specific to which affix length we are computing.
  string WorkspaceName() const override;

  // Returns the total number of affixes in the table, regardless of specified
  // length.
  FeatureValue NumValues() const override { return affix_table_->size() + 1; }

  // Special value for strings not in the map.
  FeatureValue UnknownValue() const { return affix_table_->size(); }

  // Looks up the affix for a given word.
  FeatureValue ComputeValue(const Token &token) const override;

  // Returns the string associated with a value.
  string GetFeatureValueName(FeatureValue value) const override;

 private:
  // Size parameter for the affix table.
  int affix_length_;

  // Name of the input for the table.
  string input_name_;

  // The type of the affix table.
  const AffixTable::Type type_;

  // Affix table used for indexing. This comes from the shared store, and is not
  // owned directly.
  const AffixTable *affix_table_ = nullptr;
};

// Specific instantiation for computing prefixes. This requires the input
// "prefix-table".
class PrefixFeature : public AffixTableFeature {
 public:
  PrefixFeature() : AffixTableFeature(AffixTable::PREFIX) {}
};

// Specific instantiation for computing suffixes. Requires the input
// "suffix-table."
class SuffixFeature : public AffixTableFeature {
 public:
  SuffixFeature() : AffixTableFeature(AffixTable::SUFFIX) {}
};

// Offset locator. Simple locator: just changes the focus by some offset.
class Offset : public Locator<Offset> {
 public:
  void UpdateArgs(const WorkspaceSet &workspaces,
                  const Sentence &sentence, int *focus) const {
    *focus += argument();
  }
};

typedef FeatureExtractor<Sentence, int> SentenceExtractor;

// Utility to register the sentence_instance::Feature functions.
#define REGISTER_SENTENCE_IDX_FEATURE(name, type) \
  REGISTER_FEATURE_FUNCTION(SentenceFeature, name, type)

}  // namespace syntaxnet

#endif  // $TARGETDIR_SENTENCE_FEATURES_H_
