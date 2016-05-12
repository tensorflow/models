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

#include "syntaxnet/registry.h"
#include "util/utf8/unicodetext.h"

namespace syntaxnet {

TermFrequencyMapFeature::~TermFrequencyMapFeature() {
  if (term_map_ != nullptr) {
    SharedStore::Release(term_map_);
    term_map_ = nullptr;
  }
}

void TermFrequencyMapFeature::Setup(TaskContext *context) {
  TokenLookupFeature::Setup(context);
  context->GetInput(input_name_, "text", "");
}

void TermFrequencyMapFeature::Init(TaskContext *context) {
  min_freq_ = GetIntParameter("min-freq", 0);
  max_num_terms_ = GetIntParameter("max-num-terms", 0);
  file_name_ = context->InputFile(*context->GetInput(input_name_));
  term_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
      file_name_, min_freq_, max_num_terms_);
  TokenLookupFeature::Init(context);
}

string TermFrequencyMapFeature::GetFeatureValueName(FeatureValue value) const {
  if (value == UnknownValue()) return "<UNKNOWN>";
  if (value >= 0 && value < (NumValues() - 1)) {
    return term_map_->GetTerm(value);
  }
  LOG(ERROR) << "Invalid feature value: " << value;
  return "<INVALID>";
}

string TermFrequencyMapFeature::WorkspaceName() const {
  return SharedStoreUtils::CreateDefaultName("term-frequency-map", input_name_,
                                             min_freq_, max_num_terms_);
}

string Hyphen::GetFeatureValueName(FeatureValue value) const {
  switch (value) {
    case NO_HYPHEN:
      return "NO_HYPHEN";
    case HAS_HYPHEN:
      return "HAS_HYPHEN";
  }
  return "<INVALID>";
}

FeatureValue Hyphen::ComputeValue(const Token &token) const {
  const string &word = token.word();
  return (word.find('-') < word.length() ? HAS_HYPHEN : NO_HYPHEN);
}

string Digit::GetFeatureValueName(FeatureValue value) const {
  switch (value) {
    case NO_DIGIT:
      return "NO_DIGIT";
    case SOME_DIGIT:
      return "SOME_DIGIT";
    case ALL_DIGIT:
      return "ALL_DIGIT";
  }
  return "<INVALID>";
}

FeatureValue Digit::ComputeValue(const Token &token) const {
  const string &word = token.word();
  bool has_digit = isdigit(word[0]);
  bool all_digit = has_digit;
  for (size_t i = 1; i < word.length(); ++i) {
    bool char_is_digit = isdigit(word[i]);
    all_digit = all_digit && char_is_digit;
    has_digit = has_digit || char_is_digit;
    if (!all_digit && has_digit) return SOME_DIGIT;
  }
  if (!all_digit) return NO_DIGIT;
  return ALL_DIGIT;
}

AffixTableFeature::AffixTableFeature(AffixTable::Type type)
    : type_(type) {
  if (type == AffixTable::PREFIX) {
    input_name_ = "prefix-table";
  } else {
    input_name_ = "suffix-table";
  }
}

AffixTableFeature::~AffixTableFeature() {
  SharedStore::Release(affix_table_);
  affix_table_ = nullptr;
}

string AffixTableFeature::WorkspaceName() const {
  return SharedStoreUtils::CreateDefaultName(
      "affix-table", input_name_, type_, affix_length_);
}

// Utility function to create a new affix table without changing constructors,
// to be called by the SharedStore.
static AffixTable *CreateAffixTable(const string &filename,
                                    AffixTable::Type type) {
  AffixTable *affix_table = new AffixTable(type, 1);
  tensorflow::RandomAccessFile *file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
  ProtoRecordReader reader(file);
  affix_table->Read(&reader);
  return affix_table;
}

void AffixTableFeature::Setup(TaskContext *context) {
  context->GetInput(input_name_, "recordio", "affix-table");
  affix_length_ = GetIntParameter("length", 0);
  CHECK_GE(affix_length_, 0)
      << "Length must be specified for affix preprocessor.";
  TokenLookupFeature::Setup(context);
}

void AffixTableFeature::Init(TaskContext *context) {
  string filename = context->InputFile(*context->GetInput(input_name_));

  // Get the shared AffixTable object.
  std::function<AffixTable *()> closure =
      std::bind(CreateAffixTable, filename, type_);
  affix_table_ = SharedStore::ClosureGetOrDie(filename, &closure);
  CHECK_GE(affix_table_->max_length(), affix_length_)
      << "Affixes of length " << affix_length_ << " needed, but the affix "
      <<"table only provides affixes of length <= "
      << affix_table_->max_length() << ".";
  TokenLookupFeature::Init(context);
}

FeatureValue AffixTableFeature::ComputeValue(const Token &token) const {
  const string &word = token.word();
  UnicodeText text;
  text.PointToUTF8(word.c_str(), word.size());
  if (affix_length_ > text.size()) return UnknownValue();
  UnicodeText::const_iterator start, end;
  if (type_ == AffixTable::PREFIX) {
    start = end = text.begin();
    for (int i = 0; i < affix_length_; ++i) ++end;
  } else {
    start = end = text.end();
    for (int i = 0; i < affix_length_; ++i) --start;
  }
  string affix(start.utf8_data(), end.utf8_data() - start.utf8_data());
  int affix_id = affix_table_->AffixId(affix);
  return affix_id == -1 ? UnknownValue() : affix_id;
}

string AffixTableFeature::GetFeatureValueName(FeatureValue value) const {
  if (value == UnknownValue()) return "<UNKNOWN>";
  if (value >= 0 && value < UnknownValue()) {
    return affix_table_->AffixForm(value);
  }
  LOG(ERROR) << "Invalid feature value: " << value;
  return "<INVALID>";
}

// Registry for the Sentence + token index feature functions.
REGISTER_CLASS_REGISTRY("sentence+index feature function", SentenceFeature);

// Register the features defined in the header.
REGISTER_SENTENCE_IDX_FEATURE("word", Word);
REGISTER_SENTENCE_IDX_FEATURE("lcword", LowercaseWord);
REGISTER_SENTENCE_IDX_FEATURE("tag", Tag);
REGISTER_SENTENCE_IDX_FEATURE("offset", Offset);
REGISTER_SENTENCE_IDX_FEATURE("hyphen", Hyphen);
REGISTER_SENTENCE_IDX_FEATURE("digit", Digit);
REGISTER_SENTENCE_IDX_FEATURE("prefix", PrefixFeature);
REGISTER_SENTENCE_IDX_FEATURE("suffix", SuffixFeature);

}  // namespace syntaxnet
