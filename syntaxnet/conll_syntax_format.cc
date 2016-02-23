// CoNLL document format reader for dependency annotated corpora.

#include <memory>
#include <string>
#include <vector>

#include "base/logging.h"
#include "nlp/saft/components/dependencies/opensource/document_format.h"
#include "nlp/saft/components/dependencies/opensource/sentence.pb.h"
#include "nlp/saft/components/dependencies/opensource/utils.h"
#include "third_party/tensorflow/core/lib/strings/strcat.h"
#include "third_party/tensorflow/core/lib/strings/stringprintf.h"
#include "util/regexp/re2/re2.h"

namespace neurosis {

class CoNLLSyntaxFormat : public DocumentFormat {
 public:
  virtual void ConvertFromString(const string &key,
                                 const string &value,
                                 vector<Sentence *> *documents) {
    // Create new document.
    Sentence *document = new Sentence();

    // Each line corresponds to one token.
    string text;
    vector<string> lines = utils::Split(value, '\n');

    // Add each token to the document.
    vector<string> fields;
    for (int i = 0; i < lines.size(); ++i) {
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
      for (int j = 2; j < fields.size(); ++j) {
        if (fields[j].length() == 1 && fields[j][0] == '_') fields[j].clear();
      }

      // Check that the line is valid.
      CHECK_GE(fields.size(), 7);

      // Get relevant fields.
      string &word = fields[1];
      string &cpostag = fields[3];
      string &tag = fields[4];
      int head = utils::ParseUsing<int>(fields[6], 0, utils::ParseInt32);
      string &label = fields[7];

      // Add token to document text.
      if (!text.empty()) text.append(" ");
      int start = text.size();
      int end = start + word.size() - 1;
      text.append(word);

      // Add token to document.
      Token *token = document->add_token();
      token->set_word(word);
      token->set_start(start);
      token->set_end(end);
      if (head > 0) token->set_head(head - 1);
      if (!tag.empty())  token->set_tag(tag);
      if (!cpostag.empty()) token->set_category(cpostag);
      if (!label.empty()) token->set_label(label);
    }

    if (document->token_size() > 0) {
      document->set_docid(key);
      document->set_text(text);
      documents->push_back(document);
    } else {
      // If the sentence was empty (e.g., blank lines at the beginning of a
      // file), then don't save it.
      delete document;
    }
  }

  // Converts a document to a key/value pair.
  virtual void ConvertToString(const Sentence &document,
                               string *key, string *value) {
    *key = document.docid();
    vector<string> lines;
    for (int i = 0; i < document.token_size(); ++i) {
      vector<string> fields(10);
      fields[0] = tensorflow::strings::Printf("%d", i + 1);
      fields[1] = document.token(i).word();
      fields[2] = "_";
      fields[3] = document.token(i).category();
      fields[4] = document.token(i).tag();
      fields[5] = "_";
      fields[6] =
          tensorflow::strings::Printf("%d", document.token(i).head() + 1);
      fields[7] = document.token(i).label();
      fields[8] = "_";
      fields[9] = "_";
      lines.push_back(utils::Join(fields, "\t"));
    }
    *value = tensorflow::strings::StrCat(utils::Join(lines, "\n"), "\n\n");
  }
};

REGISTER_DOCUMENT_FORMAT("conll-sentence", CoNLLSyntaxFormat);

}  // namespace neurosis
