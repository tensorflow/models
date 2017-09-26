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

#include <memory>

#include "syntaxnet/base.h"
#include "syntaxnet/document_format.h"
#include "syntaxnet/segmenter_utils.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/regexp.h"

namespace syntaxnet {

namespace {

// Reads up to the first empty line, and returns false end of file is reached.
//
// This reader is shared by CONLL and prototext formats, where records are
// separated by double newlines.
bool DoubleNewlineReadRecord(tensorflow::io::BufferedInputStream *buffer,
                             string *record) {
  string line;
  record->clear();
  tensorflow::Status status = buffer->ReadLine(&line);
  while (!line.empty() && status.ok()) {
    tensorflow::strings::StrAppend(record, line, "\n");
    status = buffer->ReadLine(&line);
  }
  return status.ok() || !record->empty();
}

}  // namespace

// CoNLL document format reader for dependency annotated corpora.
// The expected format is described e.g. at http://ilk.uvt.nl/conll/#dataformat
//
// Data should adhere to the following rules:
//   - Data files contain sentences separated by a blank line.
//   - A sentence consists of one or tokens, each one starting on a new line.
//   - A token consists of ten fields described in the table below.
//   - Fields are separated by a single tab character.
//   - All data files will contains these ten fields, although only the ID
//     column is required to contain non-dummy (i.e. non-underscore) values.
// Data files should be UTF-8 encoded (Unicode).
//
// Fields:
// 1  ID:      Token counter, starting at 1 for each new sentence and increasing
//             by 1 for every new token.
// 2  FORM:    Word form or punctuation symbol.
// 3  LEMMA:   Lemma or stem.
// 4  CPOSTAG: Coarse-grained part-of-speech tag or category.
// 5  POSTAG:  Fine-grained part-of-speech tag. Note that the same POS tag
//             cannot appear with multiple coarse-grained POS tags.
// 6  FEATS:   Unordered set of syntactic and/or morphological features.
// 7  HEAD:    Head of the current token, which is either a value of ID or '0'.
// 8  DEPREL:  Dependency relation to the HEAD.
// 9  PHEAD:   Projective head of current token.
// 10 PDEPREL: Dependency relation to the PHEAD.
//
// This CoNLL reader is compatible with the CoNLL-U format described at
//   http://universaldependencies.org/format.html
// Note that this reader skips CoNLL-U multiword tokens and empty nodes.
//
// Note on reconstruct the raw text of a sentence: the raw text is constructed
// by concatenating all words (field 2) with a intervening space between
// consecutive words.  If the last field of a token is "SpaceAfter=No", there
// would be no space between current word and the next one.
class CoNLLSyntaxFormat : public DocumentFormat {
 public:
  CoNLLSyntaxFormat() {}

  void Setup(TaskContext *context) override {
    join_category_to_pos_ = context->GetBoolParameter("join_category_to_pos");
    add_pos_as_attribute_ = context->GetBoolParameter("add_pos_as_attribute");
    serialize_morph_to_pos_ =
        context->GetBoolParameter("serialize_morph_to_pos");
  }

  // Reads up to the first empty line and returns false end of file is reached.
  bool ReadRecord(tensorflow::io::BufferedInputStream *buffer,
                  string *record) override {
    return DoubleNewlineReadRecord(buffer, record);
  }

  void ConvertFromString(const string &key, const string &value,
                         std::vector<Sentence *> *sentences) override {
    // Create new sentence.
    Sentence *sentence = new Sentence();

    // Each line corresponds to one token.
    string text;
    bool add_space_to_text = true;
    std::vector<string> lines = utils::Split(value, '\n');

    // Add each token to the sentence.
    std::vector<string> fields;
    int expected_id = 1;
    for (size_t i = 0; i < lines.size(); ++i) {
      // Split line into tab-separated fields.
      fields.clear();
      fields = utils::Split(lines[i], '\t');
      if (fields.empty()) continue;

      // Skip comment lines.
      if (fields[0][0] == '#') continue;

      // Skip CoNLLU lines for multiword tokens which are indicated by
      // hyphenated line numbers, e.g., "2-4".
      // http://universaldependencies.github.io/docs/format.html
      if (RE2::FullMatch(fields[0], "[0-9]+-[0-9]+")) continue;

      // Skip CoNLLU lines for empty tokens, indicated by decimals.
      // Introduced in v2. http://universaldependencies.org/format.html
      if (RE2::FullMatch(fields[0], "[0-9]+\\.[0-9]+")) continue;

      // Clear all optional fields equal to '_'.
      for (size_t j = 2; j < fields.size(); ++j) {
        if (fields[j].length() == 1 && fields[j][0] == '_') fields[j].clear();
      }

      // Check that the line is valid.
      CHECK_GE(fields.size(), 8)
          << "Every line has to have at least 8 tab separated fields.";

      // Check that the ids follow the expected format.
      const int id = utils::ParseUsing<int>(fields[0], 0, utils::ParseInt32);
      CHECK_EQ(expected_id++, id)
          << "Token ids start at 1 for each new sentence and increase by 1 "
          << "on each new token. Sentences are separated by an empty line.";

      // Get relevant fields.
      const string &word = fields[1];
      const string &cpostag = fields[3];
      const string &tag = fields[4];
      const string &attributes = fields[5];
      const int head = utils::ParseUsing<int>(fields[6], 0, utils::ParseInt32);
      const string &label = fields[7];

      // Add token to sentence text.
      if (!text.empty() && add_space_to_text) text.append(" ");
      const int start = text.size();
      const int end = start + word.size() - 1;
      text.append(word);
      add_space_to_text = fields[9] != "SpaceAfter=No";

      // Add token to sentence.
      Token *token = sentence->add_token();
      token->set_word(word);
      token->set_start(start);
      token->set_end(end);
      if (head > 0) token->set_head(head - 1);
      if (!tag.empty()) token->set_tag(tag);
      if (!cpostag.empty()) token->set_category(cpostag);
      if (!label.empty()) token->set_label(label);
      if (!attributes.empty()) AddMorphAttributes(attributes, token);
      if (join_category_to_pos_) JoinCategoryToPos(token);
      if (add_pos_as_attribute_) AddPosAsAttribute(token);
      if (serialize_morph_to_pos_) SerializeMorphToPos(token);
    }

    if (sentence->token_size() > 0) {
      sentence->set_docid(key);
      sentence->set_text(text);
      sentences->push_back(sentence);
    } else {
      // If the sentence was empty (e.g., blank lines at the beginning of a
      // file), then don't save it.
      delete sentence;
    }
  }

  // Converts a sentence to a key/value pair.
  void ConvertToString(const Sentence &sentence, string *key,
                       string *value) override {
    *key = sentence.docid();
    std::vector<string> lines;
    for (int i = 0; i < sentence.token_size(); ++i) {
      Token token = sentence.token(i);
      if (join_category_to_pos_) SplitCategoryFromPos(&token);
      if (add_pos_as_attribute_) RemovePosFromAttributes(&token);
      std::vector<string> fields(10);
      fields[0] = tensorflow::strings::Printf("%d", i + 1);
      fields[1] = UnderscoreIfEmpty(token.word());
      fields[2] = "_";
      fields[3] = UnderscoreIfEmpty(token.category());
      fields[4] = UnderscoreIfEmpty(token.tag());
      fields[5] = GetMorphAttributes(token);
      fields[6] = tensorflow::strings::Printf("%d", token.head() + 1);
      fields[7] = UnderscoreIfEmpty(token.label());
      fields[8] = "_";
      fields[9] = "_";
      lines.push_back(utils::Join(fields, "\t"));
    }
    *value = tensorflow::strings::StrCat(utils::Join(lines, "\n"), "\n\n");
  }

 private:
  // Replaces empty fields with an undescore.
  string UnderscoreIfEmpty(const string &field) {
    return field.empty() ? "_" : field;
  }

  // Creates a TokenMorphology object out of a list of attribute values of the
  // form: a1=v1|a2=v2|... or v1|v2|...
  void AddMorphAttributes(const string &attributes, Token *token) {
    TokenMorphology *morph =
        token->MutableExtension(TokenMorphology::morphology);
    std::vector<string> att_vals = utils::Split(attributes, '|');
    for (int i = 0; i < att_vals.size(); ++i) {
      std::vector<string> att_val = utils::SplitOne(att_vals[i], '=');

      // Format is either:
      //   1) a1=v1|a2=v2..., e.g., Czech CoNLL data, or,
      //   2) v1|v2|..., e.g., German CoNLL data.
      const std::pair<string, string> name_value =
          att_val.size() == 2 ? std::make_pair(att_val[0], att_val[1])
                              : std::make_pair(att_val[0], "on");

      // We currently don't expect an empty attribute value, but might have an
      // empty attribute name due to data input errors.
      if (name_value.second.empty()) {
        LOG(WARNING) << "Invalid attributes string: " << attributes
                     << " for token: " << token->ShortDebugString();
        continue;
      }
      if (!name_value.first.empty()) {
        TokenMorphology::Attribute *attribute = morph->add_attribute();
        attribute->set_name(name_value.first);
        attribute->set_value(name_value.second);
      }
    }
  }

  // Creates a list of attribute values of the form a1=v1|a2=v2|... or v1|v2|...
  // from a TokenMorphology object.
  string GetMorphAttributes(const Token &token) {
    const TokenMorphology &morph =
        token.GetExtension(TokenMorphology::morphology);
    if (morph.attribute_size() == 0) return "_";
    string attributes;
    for (const TokenMorphology::Attribute &attribute : morph.attribute()) {
      if (!attributes.empty()) tensorflow::strings::StrAppend(&attributes, "|");
      tensorflow::strings::StrAppend(&attributes, attribute.name());
      if (attribute.value() != "on") {
        tensorflow::strings::StrAppend(&attributes, "=", attribute.value());
      }
    }
    return attributes;
  }

  void JoinCategoryToPos(Token *token) {
    token->set_tag(
        tensorflow::strings::StrCat(token->category(), "++", token->tag()));
    token->clear_category();
  }

  void SplitCategoryFromPos(Token *token) {
    const string &tag = token->tag();
    const size_t pos = tag.find("++");
    if (pos != string::npos) {
      token->set_category(tag.substr(0, pos));
      token->set_tag(tag.substr(pos + 2));
    }
  }

  void AddPosAsAttribute(Token *token) {
    if (!token->tag().empty()) {
      TokenMorphology *morph =
          token->MutableExtension(TokenMorphology::morphology);
      TokenMorphology::Attribute *attribute = morph->add_attribute();
      attribute->set_name("fPOS");
      attribute->set_value(token->tag());
    }
  }

  void RemovePosFromAttributes(Token *token) {
    // Assumes the "fPOS" attribute, if present, is the last one.
    TokenMorphology *morph =
        token->MutableExtension(TokenMorphology::morphology);
    if (morph->attribute_size() > 0 &&
        morph->attribute().rbegin()->name() == "fPOS") {
      morph->mutable_attribute()->RemoveLast();
    }
  }

  void SerializeMorphToPos(Token *token) {
    const TokenMorphology &morph =
        token->GetExtension(TokenMorphology::morphology);
    TextFormat::Printer printer;
    printer.SetSingleLineMode(true);
    string morph_str;
    printer.PrintToString(morph, &morph_str);
    token->set_tag(morph_str);
  }

  bool join_category_to_pos_ = false;
  bool add_pos_as_attribute_ = false;
  bool serialize_morph_to_pos_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(CoNLLSyntaxFormat);
};

REGISTER_SYNTAXNET_DOCUMENT_FORMAT("conll-sentence", CoNLLSyntaxFormat);

// Reader for segmentation training data format. This reader assumes the input
// format is similar to CoNLL format but with only two fileds:
//
// Fields:
// 1  FORM:        Word form or punctuation symbol.
// 2  SPACE FLAG:  Can be either 'SPACE' or 'NO_SPACE' indicates that whether
//                 there should be a space between this word and the next one in
//                 the raw text.
//
// Examples:
// To create a training example for sentence with raw text:
//   That's a good point.
// and the corresponding gold segmentation:
//   That 's a good point .
// Then the correct input is:
// That	NO_SPACE
// 's	SPACE
// a	SPACE
// good	SPACE
// point	NO_SPACE
// .	NO_SPACE
//
// Yet another example:
// To create a training example for sentence with raw text:
//   这是一个测试
// and the corresponding gold segmentation:
//   这 是 一 个 测试
// Then the correct input is:
// 这	NO_SPACE
// 是	NO_SPACE
// 一	NO_SPACE
// 个	NO_SPACE
// 测试	NO_SPACE
class SegmentationTrainingDataFormat : public CoNLLSyntaxFormat {
 public:
  // Converts to segmentation training data by breaking those word in the input
  // tokens to utf8 character based tokens. Moreover, if a character is the
  // first char of the word in the original token, then its break level is set
  // to SPACE_BREAK to indicate that the corresponding gold transition for that
  // character token is START. Otherwise NO_BREAK to indicate MERGE.
  void ConvertFromString(const string &key, const string &value,
                         std::vector<Sentence *> *sentences) override {
    // Create new sentence.
    Sentence *sentence = new Sentence();

    // Each line corresponds to one token.
    string text;
    std::vector<string> lines = utils::Split(value, '\n');

    // Add each token to the sentence.
    std::vector<string> fields;
    for (size_t i = 0; i < lines.size(); ++i) {
      // Split line into tab-separated fields.
      fields.clear();
      fields = utils::Split(lines[i], '\t');
      if (fields.empty()) continue;

      // Skip comment lines.
      if (fields[0][0] == '#') continue;

      // Check that the line is valid.
      CHECK_GE(fields.size(), 2)
          << "Every line has to have at least 8 tab separated fields.";

      // Get relevant fields.
      const string &word = fields[0];
      CHECK(fields[1] == "SPACE" || fields[1] == "NO_SPACE")
          << "The space field can only be either 'SPACE' or 'NO_SPACE'";
      const bool space_after = fields[1] == "SPACE";

      // Add token to sentence text.
      int start = text.size();
      text.append(word);
      if (space_after && i != lines.size() - 1) {
        text.append(" ");
      }

      // Add character-based token to sentence.
      std::vector<tensorflow::StringPiece> chars;
      SegmenterUtils::GetUTF8Chars(word, &chars);
      bool is_first_char = true;
      for (auto utf8char : chars) {
        Token *char_token = sentence->add_token();
        char_token->set_word(utf8char.ToString());
        char_token->set_start(start);
        start += char_token->word().size();
        char_token->set_end(start - 1);
        char_token->set_break_level(
            is_first_char ? Token::SPACE_BREAK : Token::NO_BREAK);
        is_first_char = false;
      }

      // Add another space token.
      if (space_after) {
        Token *char_token = sentence->add_token();
        char_token->set_word(" ");
        char_token->set_start(start);
        char_token->set_end(start);
        char_token->set_break_level(Token::SPACE_BREAK);
      }
    }

    if (sentence->token_size() > 0) {
      sentence->set_docid(key);
      sentence->set_text(text);
      sentences->push_back(sentence);
    } else {
      // If the sentence was empty (e.g., blank lines at the beginning of a
      // file), then don't save it.
      delete sentence;
    }
  }
};

REGISTER_SYNTAXNET_DOCUMENT_FORMAT("segment-train-data",
                                   SegmentationTrainingDataFormat);

// Reader for tokenized text. This reader expects every sentence to be on a
// single line and tokens on that line to be separated by single spaces.
//
class TokenizedTextFormat : public DocumentFormat {
 public:
  TokenizedTextFormat() {}

  // Reads a line and returns false if end of file is reached.
  bool ReadRecord(tensorflow::io::BufferedInputStream *buffer,
                  string *record) override {
    return buffer->ReadLine(record).ok();
  }

  void ConvertFromString(const string &key, const string &value,
                         std::vector<Sentence *> *sentences) override {
    Sentence *sentence = new Sentence();
    string text;
    for (const string &word : utils::Split(value, ' ')) {
      if (word.empty()) continue;
      const int start = text.size();
      const int end = start + word.size() - 1;
      if (!text.empty()) text.append(" ");
      text.append(word);
      Token *token = sentence->add_token();
      token->set_word(word);
      token->set_start(start);
      token->set_end(end);
    }

    if (sentence->token_size() > 0) {
      sentence->set_docid(key);
      sentence->set_text(text);
      sentences->push_back(sentence);
    } else {
      // If the sentence was empty (e.g., blank lines at the beginning of a
      // file), then don't save it.
      delete sentence;
    }
  }

  void ConvertToString(const Sentence &sentence, string *key,
                       string *value) override {
    *key = sentence.docid();
    value->clear();
    for (const Token &token : sentence.token()) {
      if (!value->empty()) value->append(" ");
      value->append(token.word());
      if (token.has_tag()) {
        value->append("_");
        value->append(token.tag());
      }
      if (token.has_head()) {
        value->append("_");
        value->append(tensorflow::strings::StrCat(token.head()));
      }
    }
    value->append("\n");
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TokenizedTextFormat);
};

REGISTER_SYNTAXNET_DOCUMENT_FORMAT("tokenized-text", TokenizedTextFormat);

// Reader for un-tokenized text. This reader expects every sentence to be on a
// single line. For each line in the input, a sentence proto will be created,
// where tokens are utf8 characters of that line.
//
class UntokenizedTextFormat : public TokenizedTextFormat {
 public:
  UntokenizedTextFormat() {}

  void ConvertFromString(const string &key, const string &value,
                         std::vector<Sentence *> *sentences) override {
    Sentence *sentence = new Sentence();
    std::vector<tensorflow::StringPiece> chars;
    SegmenterUtils::GetUTF8Chars(value, &chars);
    int start = 0;
    for (auto utf8char : chars) {
      Token *token = sentence->add_token();
      token->set_word(utf8char.ToString());
      token->set_start(start);
      start += utf8char.size();
      token->set_end(start - 1);
    }

    if (sentence->token_size() > 0) {
      sentence->set_docid(key);
      sentence->set_text(value);
      sentences->push_back(sentence);
    } else {
      // If the sentence was empty (e.g., blank lines at the beginning of a
      // file), then don't save it.
      delete sentence;
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(UntokenizedTextFormat);
};

REGISTER_SYNTAXNET_DOCUMENT_FORMAT("untokenized-text", UntokenizedTextFormat);

// Text reader that attmpts to perform Penn Treebank tokenization on arbitrary
// raw text. Adapted from https://www.cis.upenn.edu/~treebank/tokenizer.sed
// by Robert MacIntyre, University of Pennsylvania, late 1995.
// Expected input: raw text with one sentence per line.
//
class EnglishTextFormat : public TokenizedTextFormat {
 public:
  EnglishTextFormat() {}

  void ConvertFromString(const string &key, const string &value,
                         std::vector<Sentence *> *sentences) override {
    std::vector<std::pair<string, string>> preproc_rules = {
        // Punctuation.
        {"’", "'"},
        {"…", "..."},
        {"---", "--"},
        {"—", "--"},
        {"–", "--"},
        {"，", ","},
        {"。", "."},
        {"！", "!"},
        {"？", "?"},
        {"：", ":"},
        {"；", ";"},
        {"＆", "&"},

        // Brackets.
        {"\\[", "("},
        {"]", ")"},
        {"{", "("},
        {"}", ")"},
        {"【", "("},
        {"】", ")"},
        {"（", "("},
        {"）", ")"},

        // Quotation marks.
        {"\"", "\""},
        {"″", "\""},
        {"“", "\""},
        {"„", "\""},
        {"‵‵", "\""},
        {"”", "\""},
        {"’", "\""},
        {"‘", "\""},
        {"′′", "\""},
        {"‹", "\""},
        {"›", "\""},
        {"«", "\""},
        {"»", "\""},

        // Discarded punctuation that breaks sentences.
        {"|", ""},
        {"·", ""},
        {"•", ""},
        {"●", ""},
        {"▪", ""},
        {"■", ""},
        {"□", ""},
        {"❑", ""},
        {"◆", ""},
        {"★", ""},
        {"＊", ""},
        {"♦", ""},
    };

    std::vector<std::pair<string, string>> rules = {
        // attempt to get correct directional quotes
        {R"re(^")re", "`` "},
        {R"re(([ \([{<])")re", "\\1 `` "},
        // close quotes handled at end

        {R"re(\.\.\.)re", " ... "},
        {"[,;:@#$%&]", " \\0 "},

        // Assume sentence tokenization has been done first, so split FINAL
        // periods only.
        {R"re(([^.])(\.)([\]\)}>"']*)[ ]*$)re", "\\1 \\2\\3 "},
        // however, we may as well split ALL question marks and exclamation
        // points, since they shouldn't have the abbrev.-marker ambiguity
        // problem
        {"[?!]", " \\0 "},

        // parentheses, brackets, etc.
        {R"re([\]\[\(\){}<>])re", " \\0 "},

        // Like Adwait Ratnaparkhi's MXPOST, we use the parsed-file version of
        // these symbols.
        {"\\(", "-LRB-"},
        {"\\)", "-RRB-"},
        {"\\]", "-LSB-"},
        {"\\]", "-RSB-"},
        {"{", "-LCB-"},
        {"}", "-RCB-"},

        {"--", " -- "},

        // First off, add a space to the beginning and end of each line, to
        // reduce necessary number of regexps.
        {"$", " "},
        {"^", " "},

        {"\"", " '' "},
        // possessive or close-single-quote
        {"([^'])' ", "\\1 ' "},
        // as in it's, I'm, we'd
        {"'([sSmMdD]) ", " '\\1 "},
        {"'ll ", " 'll "},
        {"'re ", " 're "},
        {"'ve ", " 've "},
        {"n't ", " n't "},
        {"'LL ", " 'LL "},
        {"'RE ", " 'RE "},
        {"'VE ", " 'VE "},
        {"N'T ", " N'T "},

        {" ([Cc])annot ", " \\1an not "},
        {" ([Dd])'ye ", " \\1' ye "},
        {" ([Gg])imme ", " \\1im me "},
        {" ([Gg])onna ", " \\1on na "},
        {" ([Gg])otta ", " \\1ot ta "},
        {" ([Ll])emme ", " \\1em me "},
        {" ([Mm])ore'n ", " \\1ore 'n "},
        {" '([Tt])is ", " '\\1 is "},
        {" '([Tt])was ", " '\\1 was "},
        {" ([Ww])anna ", " \\1an na "},
        {" ([Ww])haddya ", " \\1ha dd ya "},
        {" ([Ww])hatcha ", " \\1ha t cha "},

        // clean out extra spaces
        {"  *", " "},
        {"^ *", ""},
    };

    string rewritten = value;
    for (const std::pair<string, string> &rule : preproc_rules) {
      RE2::GlobalReplace(&rewritten, rule.first, rule.second);
    }
    for (const std::pair<string, string> &rule : rules) {
      RE2::GlobalReplace(&rewritten, rule.first, rule.second);
    }
    TokenizedTextFormat::ConvertFromString(key, rewritten, sentences);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EnglishTextFormat);
};

REGISTER_SYNTAXNET_DOCUMENT_FORMAT("english-text", EnglishTextFormat);

// Converts double-newline-separated prototext records into sentences.
class SentencePrototextFormat : public DocumentFormat {
 public:
  SentencePrototextFormat() {}

  bool ReadRecord(tensorflow::io::BufferedInputStream *buffer,
                  string *record) override {
    return DoubleNewlineReadRecord(buffer, record);
  }

  void ConvertFromString(const string &key, const string &value,
                         std::vector<Sentence *> *sentences) override {
    Sentence *sentence = new Sentence();
    CHECK(TextFormat::ParseFromString(value, sentence))
        << "Failed to parse " << value;
    sentences->push_back(sentence);
  }

  void ConvertToString(const Sentence &sentence, string *key,
                       string *value) override {
    *key = sentence.docid();
    string as_prototext;
    CHECK(TextFormat::PrintToString(sentence, &as_prototext))
        << "Failed to sentence with ID " << (*key);
    *value = tensorflow::strings::StrCat(as_prototext, "\n\n");
  }
};

REGISTER_SYNTAXNET_DOCUMENT_FORMAT("sentence-prototext",
                                   SentencePrototextFormat);

}  // namespace syntaxnet
