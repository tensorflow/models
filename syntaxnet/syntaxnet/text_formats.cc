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
#include <string>
#include <vector>

#include "syntaxnet/document_format.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/regexp.h"

namespace syntaxnet {

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
// Note that this reader skips CoNLL-U multiword tokens and ignores the last two
// fields of every line, which are PHEAD and PDEPREL in CoNLL format, but are
// replaced by DEPS and MISC in CoNLL-U.
//
class CoNLLSyntaxFormat : public DocumentFormat {
 public:
  CoNLLSyntaxFormat() {}

  // Reads up to the first empty line and returns false end of file is reached.
  bool ReadRecord(tensorflow::io::InputBuffer *buffer,
                  string *record) override {
    string line;
    record->clear();
    tensorflow::Status status = buffer->ReadLine(&line);
    while (!line.empty() && status.ok()) {
      tensorflow::strings::StrAppend(record, line, "\n");
      status = buffer->ReadLine(&line);
    }
    return status.ok() || !record->empty();
  }

  void ConvertFromString(const string &key, const string &value,
                         vector<Sentence *> *sentences) override {
    // Create new sentence.
    Sentence *sentence = new Sentence();

    // Each line corresponds to one token.
    string text;
    vector<string> lines = utils::Split(value, '\n');

    // Add each token to the sentence.
    vector<string> fields;
    int expected_id = 1;
    for (size_t i = 0; i < lines.size(); ++i) {
      // Split line into tab-separated fields.
      fields.clear();
      fields = utils::Split(lines[i], '\t');
      if (fields.size() == 0) continue;

      // Skip comment lines.
      if (fields[0][0] == '#') continue;

      // Skip CoNLLU lines for multiword tokens which are indicated by
      // hyphenated line numbers, e.g., "2-4".
      // http://universaldependencies.github.io/docs/format.html
      if (RE2::FullMatch(fields[0], "[0-9]+-[0-9]+")) continue;

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
      const int head = utils::ParseUsing<int>(fields[6], 0, utils::ParseInt32);
      const string &label = fields[7];

      // Add token to sentence text.
      if (!text.empty()) text.append(" ");
      const int start = text.size();
      const int end = start + word.size() - 1;
      text.append(word);

      // Add token to sentence.
      Token *token = sentence->add_token();
      token->set_word(word);
      token->set_start(start);
      token->set_end(end);
      if (head > 0) token->set_head(head - 1);
      if (!tag.empty()) token->set_tag(tag);
      if (!cpostag.empty()) token->set_category(cpostag);
      if (!label.empty()) token->set_label(label);
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
    vector<string> lines;
    for (int i = 0; i < sentence.token_size(); ++i) {
      vector<string> fields(10);
      fields[0] = tensorflow::strings::Printf("%d", i + 1);
      fields[1] = sentence.token(i).word();
      fields[2] = "_";
      fields[3] = sentence.token(i).category();
      fields[4] = sentence.token(i).tag();
      fields[5] = "_";
      fields[6] =
          tensorflow::strings::Printf("%d", sentence.token(i).head() + 1);
      fields[7] = sentence.token(i).label();
      fields[8] = "_";
      fields[9] = "_";
      lines.push_back(utils::Join(fields, "\t"));
    }
    *value = tensorflow::strings::StrCat(utils::Join(lines, "\n"), "\n\n");
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CoNLLSyntaxFormat);
};

REGISTER_DOCUMENT_FORMAT("conll-sentence", CoNLLSyntaxFormat);

// Reader for tokenized text. This reader expects every sentence to be on a
// single line and tokens on that line to be separated by single spaces.
//
class TokenizedTextFormat : public DocumentFormat {
 public:
  TokenizedTextFormat() {}

  // Reads a line and returns false if end of file is reached.
  bool ReadRecord(tensorflow::io::InputBuffer *buffer,
                  string *record) override {
    return buffer->ReadLine(record).ok();
  }

  void ConvertFromString(const string &key, const string &value,
                         vector<Sentence *> *sentences) override {
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

REGISTER_DOCUMENT_FORMAT("tokenized-text", TokenizedTextFormat);

// Text reader that attmpts to perform Penn Treebank tokenization on arbitrary
// raw text. Adapted from https://www.cis.upenn.edu/~treebank/tokenizer.sed
// by Robert MacIntyre, University of Pennsylvania, late 1995.
// Expected input: raw text with one sentence per line.
//
class EnglishTextFormat : public TokenizedTextFormat {
 public:
  EnglishTextFormat() {}

  void ConvertFromString(const string &key, const string &value,
                         vector<Sentence *> *sentences) override {
    vector<pair<string, string>> preproc_rules = {
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

    vector<pair<string, string>> rules = {
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
    for (const pair<string, string> &rule : preproc_rules) {
      RE2::GlobalReplace(&rewritten, rule.first, rule.second);
    }
    for (const pair<string, string> &rule : rules) {
      RE2::GlobalReplace(&rewritten, rule.first, rule.second);
    }
    TokenizedTextFormat::ConvertFromString(key, rewritten, sentences);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(EnglishTextFormat);
};

REGISTER_DOCUMENT_FORMAT("english-text", EnglishTextFormat);

}  // namespace syntaxnet
