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

#ifndef SYNTAXNET_SENTENCE_FEATURES_H_
#define SYNTAXNET_SENTENCE_FEATURES_H_

#include "syntaxnet/affix.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/feature_types.h"
#include "syntaxnet/segmenter_utils.h"
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

  int Workspace() const { return workspace_; }

 private:
  int workspace_;
};

// A multi purpose specialization of the feature. Processes the tokens in a
// Sentence by looking up a value set for each token and storing that in
// a VectorVectorInt workspace. Given a set of base values of size Size(),
// reserves an extra value for unknown tokens.
class TokenLookupSetFeature : public SentenceFeature {
 public:
  void Init(TaskContext *context) override {
    set_feature_type(new ResourceBasedFeatureType<TokenLookupSetFeature>(
        name(), this, {{NumValues(), "<OUTSIDE>"}}));
  }

  // Number of unique values.
  virtual int64 NumValues() const = 0;

  // Given a position in a sentence and workspaces, looks up the corresponding
  // feature value set. The index is relative to the start of the sentence.
  virtual void LookupToken(const WorkspaceSet &workspaces,
                           const Sentence &sentence, int index,
                           std::vector<int> *values) const = 0;

  // Given a feature value, returns a string representation.
  virtual string GetFeatureValueName(int value) const = 0;

  // Name of the shared workspace.
  virtual string WorkspaceName() const = 0;

  // TokenLookupSetFeatures use VectorVectorIntWorkspaces by default.
  void RequestWorkspaces(WorkspaceRegistry *registry) override {
    workspace_ = registry->Request<VectorVectorIntWorkspace>(WorkspaceName());
  }

  // Default preprocessing: looks up a value set for each token in the Sentence.
  void Preprocess(WorkspaceSet *workspaces, Sentence *sentence) const override {
    // Default preprocessing: lookup a value set for each token in the Sentence.
    if (workspaces->Has<VectorVectorIntWorkspace>(workspace_)) return;
    VectorVectorIntWorkspace *workspace =
        new VectorVectorIntWorkspace(sentence->token_size());
    for (int i = 0; i < sentence->token_size(); ++i) {
      LookupToken(*workspaces, *sentence, i, workspace->mutable_elements(i));
    }
    workspaces->Set<VectorVectorIntWorkspace>(workspace_, workspace);
  }

  // Returns a pre-computed token value from the cache. This assumes the cache
  // is populated.
  const std::vector<int> &GetCachedValueSet(const WorkspaceSet &workspaces,
                                       const Sentence &sentence,
                                       int focus) const {
    // Do bounds checking on focus.
    CHECK_GE(focus, 0);
    CHECK_LT(focus, sentence.token_size());

    // Return value from cache.
    return workspaces.Get<VectorVectorIntWorkspace>(workspace_).elements(focus);
  }

  // Adds any precomputed features at the given focus, if present.
  void Evaluate(const WorkspaceSet &workspaces, const Sentence &sentence,
                int focus, FeatureVector *result) const override {
    if (focus >= 0 && focus < sentence.token_size()) {
      const std::vector<int> &elements =
          GetCachedValueSet(workspaces, sentence, focus);
      for (auto &value : elements) {
        result->add(this->feature_type(), value);
      }
    }
  }

  // Returns the precomputed value, or NumValues() for features outside
  // the sentence.
  FeatureValue Compute(const WorkspaceSet &workspaces, const Sentence &sentence,
                       int focus, const FeatureVector *result) const override {
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
  int64 NumValues() const override { return term_map_->Size() + 1; }

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

// Specialization of the TokenLookupSetFeature class to use a TermFrequencyMap
// to perform the mapping. This takes two options: "min_freq" (discard tokens
// with less than this min frequency), and "max_num_terms" (only read in at most
// these terms.)
class TermFrequencyMapSetFeature : public TokenLookupSetFeature {
 public:
  // Initializes with an empty name, since we need the options to compute the
  // actual workspace name.
  explicit TermFrequencyMapSetFeature(const string &input_name)
      : input_name_(input_name), min_freq_(0), max_num_terms_(0) {}

  // Releases shared resources.
  ~TermFrequencyMapSetFeature() override;

  // Returns index of raw word text.
  virtual void GetTokenIndices(const Token &token,
                               std::vector<int> *values) const = 0;

  // Requests the resource inputs.
  void Setup(TaskContext *context) override;

  // Obtains resources using the shared store. At this point options are known
  // so the full name can be computed.
  void Init(TaskContext *context) override;

  // Number of unique values.

  int64 NumValues() const override { return term_map_->Size(); }

  // Special value for strings not in the map.
  FeatureValue UnknownValue() const { return term_map_->Size(); }

  // Gets pointer to the underlying map.
  const TermFrequencyMap *term_map() const { return term_map_; }

  // Returns the term index or the unknown value. Used inside GetTokenIndex()
  // specializations for convenience.
  int LookupIndex(const string &term) const {
    return term_map_->LookupIndex(term, -1);
  }

  // Given a position in a sentence and workspaces, looks up the corresponding
  // feature value set. The index is relative to the start of the sentence.
  void LookupToken(const WorkspaceSet &workspaces, const Sentence &sentence,
                   int index, std::vector<int> *values) const override {
    GetTokenIndices(sentence.token(index), values);
  }

  // Uses the TermFrequencyMap to lookup the string associated with a value.
  string GetFeatureValueName(int value) const override {
    if (value == UnknownValue()) return "<UNKNOWN>";
    if (value >= 0 && value < NumValues()) {
      return term_map_->GetTerm(value);
    }
    LOG(ERROR) << "Invalid feature value: " << value;
    return "<INVALID>";
  }

  // Name of the shared workspace.
  string WorkspaceName() const override;

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
    const string &form = token.word();
    return term_map().LookupIndex(form, UnknownValue());
  }
};

class Char : public TermFrequencyMapFeature {
 public:
  Char() : TermFrequencyMapFeature("char-map") {}

  FeatureValue ComputeValue(const Token &token) const override {
    const string &form = token.word();
    if (SegmenterUtils::IsBreakChar(form)) return BreakCharValue();
    return term_map().LookupIndex(form, UnknownValue());
  }

  // Special value for breaks.
  FeatureValue BreakCharValue() const { return term_map().Size(); }

  // Special value for non-break strings not in the map.
  FeatureValue UnknownValue() const { return term_map().Size() + 1; }

  // Number of unique values.
  int64 NumValues() const override { return term_map().Size() + 2; }

  string GetFeatureValueName(FeatureValue value) const override {
    if (value == BreakCharValue()) return "<BREAK_CHAR>";
    if (value == UnknownValue()) return "<UNKNOWN>";
    if (value >= 0 && value < term_map().Size()) {
      return term_map().GetTerm(value);
    }
    LOG(ERROR) << "Invalid feature value: " << value;
    return "<INVALID>";
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

class CharNgram : public TermFrequencyMapSetFeature {
 public:
  CharNgram() : TermFrequencyMapSetFeature("char-ngram-map") {}
  ~CharNgram() override {}

  void Setup(TaskContext *context) override {
    TermFrequencyMapSetFeature::Setup(context);
    max_char_ngram_length_ = context->Get("lexicon_max_char_ngram_length", 3);
    use_terminators_ =
        context->Get("lexicon_char_ngram_include_terminators", false);
  }

  // Returns index of raw word text.
  void GetTokenIndices(const Token &token,
                       std::vector<int> *values) const override;

 private:
  // Size parameter (n) for the ngrams.
  int max_char_ngram_length_ = 3;

  // Whether to pad the word with ^ and $ before extracting ngrams.
  bool use_terminators_ = false;
};

class MorphologySet : public TermFrequencyMapSetFeature {
 public:
  MorphologySet() : TermFrequencyMapSetFeature("morphology-map") {}
  ~MorphologySet() override {}

  void Setup(TaskContext *context) override {
    TermFrequencyMapSetFeature::Setup(context);
  }


  int64 NumValues() const override {
    return term_map()->Size() - 1;
  }

  // Returns index of raw word text.
  void GetTokenIndices(const Token &token,
                       std::vector<int> *values) const override;
};

class LexicalCategoryFeature : public TokenLookupFeature {
 public:
  LexicalCategoryFeature(const string &name, int cardinality)
      : name_(name), cardinality_(cardinality) {}
  ~LexicalCategoryFeature() override {}

  FeatureValue NumValues() const override { return cardinality_; }

  // Returns the identifier for the workspace for this feature.
  string WorkspaceName() const override {
    return tensorflow::strings::StrCat(name_, ":", cardinality_);
  }

 private:
  // Name of the category type.
  const string name_;

  // Number of values.
  const int cardinality_;
};

// Feature that computes whether a word has a hyphen or not.
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

// Feature that categorizes the capitalization of the word. If the option
// utf8=true is specified, lowercase and uppercase checks are done with UTF8
// compliant functions.
class Capitalization : public LexicalCategoryFeature {
 public:
  // Enumeration of values.
  enum Category {
    LOWERCASE = 0,                     // normal word
    UPPERCASE = 1,                     // all-caps
    CAPITALIZED = 2,                   // has one cap and one non-cap
    CAPITALIZED_SENTENCE_INITIAL = 3,  // same as above but sentence-initial
    NON_ALPHABETIC = 4,                // contains no alphabetic characters
    CARDINALITY = 5,
  };

  // Default constructor.
  Capitalization() : LexicalCategoryFeature("capitalization", CARDINALITY) {}

  // Sets one of the options for the capitalization.
  void Setup(TaskContext *context) override;

  // Capitalization needs special preprocessing because token category can
  // depend on whether the token is at the start of the sentence.
  void Preprocess(WorkspaceSet *workspaces, Sentence *sentence) const override;

  // Returns a string representation of the enum value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the category value for the token.
  FeatureValue ComputeValue(const Token &token) const override {
    LOG(FATAL) << "Capitalization should use ComputeValueWithFocus.";
    return 0;
  }

  // Returns the category value for the token.
  FeatureValue ComputeValueWithFocus(const Token &token, int focus) const;

 private:
  // Whether to use UTF8 compliant functions to check capitalization.
  bool utf8_ = false;
};

// A feature for computing whether the focus token contains any punctuation
// for ternary features.
class PunctuationAmount : public LexicalCategoryFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_PUNCTUATION = 0,
    SOME_PUNCTUATION = 1,
    ALL_PUNCTUATION = 2,
    CARDINALITY = 3,
  };

  // Default constructor.
  PunctuationAmount()
      : LexicalCategoryFeature("punctuation-amount", CARDINALITY) {}

  // Returns a string representation of the enum value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the category value for the token.
  FeatureValue ComputeValue(const Token &token) const override;
};

// A feature for a feature that returns whether the word is an open or
// close quotation mark, based on its relative position to other quotation marks
// in the sentence.
class Quote : public LexicalCategoryFeature {
 public:
  // Enumeration of values.
  enum Category {
    NO_QUOTE = 0,
    OPEN_QUOTE = 1,
    CLOSE_QUOTE = 2,
    UNKNOWN_QUOTE = 3,
    CARDINALITY = 4,
  };

  // Default constructor.
  Quote() : LexicalCategoryFeature("quote", CARDINALITY) {}

  // Returns a string representation of the enum value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the category value for the token.
  FeatureValue ComputeValue(const Token &token) const override;

  // Override preprocess to compute open and close quotes from prior context of
  // the sentence.
  void Preprocess(WorkspaceSet *workspaces, Sentence *instance) const override;
};

// Feature that computes whether a word has digits or not.
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

// TokenLookupFeature object to compute prefixes and suffixes of words. The
// AffixTable is stored in the SharedStore. This is very similar to the
// implementation of TermFrequencyMapFeature, but using an AffixTable to
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
  REGISTER_SYNTAXNET_FEATURE_FUNCTION(SentenceFeature, name, type)

}  // namespace syntaxnet

#endif  // SYNTAXNET_SENTENCE_FEATURES_H_
