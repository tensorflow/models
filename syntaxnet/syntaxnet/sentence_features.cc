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
#include "syntaxnet/char_properties.h"
#include "syntaxnet/registry.h"
#include "util/utf8/unicodetext.h"
#include "util/utf8/unilib.h"
#include "util/utf8/unilib_utf8_utils.h"

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

TermFrequencyMapSetFeature::~TermFrequencyMapSetFeature() {
  if (term_map_ != nullptr) {
    SharedStore::Release(term_map_);
    term_map_ = nullptr;
  }
}

void TermFrequencyMapSetFeature::Setup(TaskContext *context) {
  context->GetInput(input_name_, "text", "");
}

void TermFrequencyMapSetFeature::Init(TaskContext *context) {
  min_freq_ = GetIntParameter("min-freq", 0);
  max_num_terms_ = GetIntParameter("max-num-terms", 0);
  file_name_ = context->InputFile(*context->GetInput(input_name_));
  term_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
      file_name_, min_freq_, max_num_terms_);
  TokenLookupSetFeature::Init(context);
}

string TermFrequencyMapSetFeature::WorkspaceName() const {
  return SharedStoreUtils::CreateDefaultName(
      "term-frequency-map-set", input_name_, min_freq_, max_num_terms_);
}

namespace {
void GetUTF8Chars(const string &word,
                  std::vector<tensorflow::StringPiece> *chars) {
  UnicodeText text;
  text.PointToUTF8(word.c_str(), word.size());
  for (UnicodeText::const_iterator it = text.begin(); it != text.end(); ++it) {
    chars->push_back(tensorflow::StringPiece(it.utf8_data(), it.utf8_length()));
  }
}

int UTF8FirstLetterNumBytes(const char *utf8_str) {
  if (*utf8_str == '\0') return 0;
  return UniLib::OneCharLen(utf8_str);
}

}  // namespace

void CharNgram::GetTokenIndices(const Token &token,
                                std::vector<int> *values) const {
  values->clear();
  std::vector<tensorflow::StringPiece> char_sp;
  if (use_terminators_) char_sp.push_back("^");
  GetUTF8Chars(token.word(), &char_sp);
  if (use_terminators_) char_sp.push_back("$");
  for (int start = 0; start < char_sp.size(); ++start) {
    string char_ngram;
    for (int index = 0;
         index < max_char_ngram_length_ && start + index < char_sp.size();
         ++index) {
      tensorflow::StringPiece c = char_sp[start + index];
      if (c == " ") break;  // Never add char ngrams containing spaces.
      tensorflow::strings::StrAppend(&char_ngram, c);
      int value = LookupIndex(char_ngram);
      if (value != -1) {  // Skip unknown values.
        values->push_back(value);
      }
    }
  }
}

void MorphologySet::GetTokenIndices(const Token &token,
                                    std::vector<int> *values) const {
  values->clear();
  const TokenMorphology &token_morphology =
      token.GetExtension(TokenMorphology::morphology);
  for (const TokenMorphology::Attribute &att : token_morphology.attribute()) {
    int value =
        LookupIndex(tensorflow::strings::StrCat(att.name(), "=", att.value()));
    if (value != -1) {  // Skip unknown values.
      values->push_back(value);
    }
  }
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

void Capitalization::Setup(TaskContext *context) {
  utf8_ = (GetParameter("utf8") == "true");
}

// Runs ComputeValue for each token in the sentence.
void Capitalization::Preprocess(WorkspaceSet *workspaces,
                                Sentence *sentence) const {
  if (workspaces->Has<VectorIntWorkspace>(Workspace())) return;
  VectorIntWorkspace *workspace =
      new VectorIntWorkspace(sentence->token_size());
  for (int i = 0; i < sentence->token_size(); ++i) {
    const int value = ComputeValueWithFocus(sentence->token(i), i);
    workspace->set_element(i, value);
  }
  workspaces->Set<VectorIntWorkspace>(Workspace(), workspace);
}

string Capitalization::GetFeatureValueName(FeatureValue value) const {
  switch (value) {
    case LOWERCASE:
      return "LOWERCASE";
    case UPPERCASE:
      return "UPPERCASE";
    case CAPITALIZED:
      return "CAPITALIZED";
    case CAPITALIZED_SENTENCE_INITIAL:
      return "CAPITALIZED_SENTENCE_INITIAL";
    case NON_ALPHABETIC:
      return "NON_ALPHABETIC";
  }
  return "<INVALID>";
}

FeatureValue Capitalization::ComputeValueWithFocus(const Token &token,
                                                   int focus) const {
  const string &word = token.word();

  // Check whether there is an uppercase or lowercase character.
  bool has_upper = false;
  bool has_lower = false;
  if (utf8_) {
    LOG(FATAL) << "Not implemented.";
  } else {
    const char *str = word.c_str();
    for (int i = 0; i < word.length(); ++i) {
      const char c = str[i];
      has_upper = (has_upper || (c >= 'A' && c <= 'Z'));
      has_lower = (has_lower || (c >= 'a' && c <= 'z'));
    }
  }

  // Compute simple values.
  if (!has_upper && has_lower) return LOWERCASE;
  if (has_upper && !has_lower) return UPPERCASE;
  if (!has_upper && !has_lower) return NON_ALPHABETIC;

  // Else has_upper && has_lower; a normal capitalized word.  Check the break
  // level to determine whether the capitalized word is sentence-initial.
  const bool sentence_initial = (focus == 0);
  return sentence_initial ? CAPITALIZED_SENTENCE_INITIAL : CAPITALIZED;
}

string PunctuationAmount::GetFeatureValueName(FeatureValue value) const {
  switch (value) {
    case NO_PUNCTUATION:
      return "NO_PUNCTUATION";
    case SOME_PUNCTUATION:
      return "SOME_PUNCTUATION";
    case ALL_PUNCTUATION:
      return "ALL_PUNCTUATION";
  }
  return "<INVALID>";
}

FeatureValue PunctuationAmount::ComputeValue(const Token &token) const {
  const string &word = token.word();
  bool has_punctuation = false;
  bool all_punctuation = true;

  const char *start = word.c_str();
  const char *end = word.c_str() + word.size();
  while (start < end) {
    int char_length = UTF8FirstLetterNumBytes(start);
    bool char_is_punct = is_punctuation_or_symbol(start, char_length);
    all_punctuation &= char_is_punct;
    has_punctuation |= char_is_punct;
    if (!all_punctuation && has_punctuation) return SOME_PUNCTUATION;
    start += char_length;
  }
  if (!all_punctuation) return NO_PUNCTUATION;
  return ALL_PUNCTUATION;
}

string Quote::GetFeatureValueName(FeatureValue value) const {
  switch (value) {
    case NO_QUOTE:
      return "NO_QUOTE";
    case OPEN_QUOTE:
      return "OPEN_QUOTE";
    case CLOSE_QUOTE:
      return "CLOSE_QUOTE";
    case UNKNOWN_QUOTE:
      return "UNKNOWN_QUOTE";
  }
  return "<INVALID>";
}

FeatureValue Quote::ComputeValue(const Token &token) const {
  const string &word = token.word();

  // Penn Treebank open and close quotes are multi-character.
  if (word == "``") return OPEN_QUOTE;
  if (word == "''") return CLOSE_QUOTE;
  if (word.length() == 1) {
    int char_len = UTF8FirstLetterNumBytes(word.c_str());
    bool is_open = is_open_quote(word.c_str(), char_len);
    bool is_close = is_close_quote(word.c_str(), char_len);
    if (is_open && !is_close) return OPEN_QUOTE;
    if (is_close && !is_open) return CLOSE_QUOTE;
    if (is_open && is_close) return UNKNOWN_QUOTE;
  }
  return NO_QUOTE;
}

void Quote::Preprocess(WorkspaceSet *workspaces, Sentence *sentence) const {
  if (workspaces->Has<VectorIntWorkspace>(Workspace())) return;
  VectorIntWorkspace *workspace =
      new VectorIntWorkspace(sentence->token_size());

  // For double quote ", it is unknown whether they are open or closed without
  // looking at the prior tokens in the sentence.  in_quote is true iff an odd
  // number of " marks have been seen so far in the sentence (similar to the
  // behavior of some tokenizers).
  bool in_quote = false;
  for (int i = 0; i < sentence->token_size(); ++i) {
    int quote_type = ComputeValue(sentence->token(i));
    if (quote_type == UNKNOWN_QUOTE) {
      // Update based on in_quote and flip in_quote.
      quote_type = in_quote ? CLOSE_QUOTE : OPEN_QUOTE;
      in_quote = !in_quote;
    }
    workspace->set_element(i, quote_type);
  }
  workspaces->Set<VectorIntWorkspace>(Workspace(), workspace);
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
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
  ProtoRecordReader reader(file.release());
  affix_table->Read(&reader);
  return affix_table;
}

void AffixTableFeature::Setup(TaskContext *context) {
  context->GetInput(input_name_, "recordio", "affix-table");
  affix_length_ = GetIntParameter("length", 0);
  CHECK_GE(affix_length_, 0) << "Length must be specified for affix feature.";
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
REGISTER_SYNTAXNET_CLASS_REGISTRY("sentence+index feature function",
                                  SentenceFeature);

// Register the features defined in the header.
REGISTER_SENTENCE_IDX_FEATURE("word", Word);
REGISTER_SENTENCE_IDX_FEATURE("char", Char);
REGISTER_SENTENCE_IDX_FEATURE("lcword", LowercaseWord);
REGISTER_SENTENCE_IDX_FEATURE("tag", Tag);
REGISTER_SENTENCE_IDX_FEATURE("offset", Offset);
REGISTER_SENTENCE_IDX_FEATURE("hyphen", Hyphen);
REGISTER_SENTENCE_IDX_FEATURE("digit", Digit);
REGISTER_SENTENCE_IDX_FEATURE("prefix", PrefixFeature);
REGISTER_SENTENCE_IDX_FEATURE("suffix", SuffixFeature);
REGISTER_SENTENCE_IDX_FEATURE("char-ngram", CharNgram);
REGISTER_SENTENCE_IDX_FEATURE("morphology-set", MorphologySet);
REGISTER_SENTENCE_IDX_FEATURE("capitalization", Capitalization);
REGISTER_SENTENCE_IDX_FEATURE("punctuation-amount", PunctuationAmount);
REGISTER_SENTENCE_IDX_FEATURE("quote", Quote);

}  // namespace syntaxnet
