#include "nlp/saft/components/dependencies/opensource/populate_test_inputs.h"

#include <map>
#include <utility>

#include "base/commandlineflags_declare.h"
#include "base/logging.h"
#include "net/proto2/public/text_format.h"
#include "nlp/saft/components/dependencies/opensource/dictionary.proto.h"
#include "nlp/saft/components/dependencies/opensource/sentence.proto.h"
#include "nlp/saft/components/dependencies/opensource/task_context.h"
#include "nlp/saft/components/dependencies/opensource/task_spec.proto.h"
#include "nlp/saft/components/dependencies/opensource/term_frequency_map.h"
#include "third_party/tensorflow/core/lib/core/status.h"
#include "third_party/tensorflow/core/lib/io/record_writer.h"
#include "third_party/tensorflow/core/lib/strings/strcat.h"
#include "third_party/tensorflow/core/platform/env.h"

DECLARE_string(test_tmpdir);

namespace neurosis {

void PopulateTestInputs::CreatorMap::Add(
    const string &name, const string &file_format, const string &record_format,
    PopulateTestInputs::CreateFile makefile) {
  (*this)[name] = [name, file_format, record_format,
                   makefile](TaskInput *input) {
    makefile(AddPart(input, file_format, record_format));
  };
}

bool PopulateTestInputs::CreatorMap::Populate(TaskContext *context) const {
  return PopulateTestInputs::Populate(*this, context);
}

PopulateTestInputs::CreatorMap PopulateTestInputs::Defaults(
    const Sentence &document) {
  CreatorMap creators;
  creators["category-map"] =
      CreateTFMapFromDocumentTokens(document, TokenCategory);
  creators["label-map"] = CreateTFMapFromDocumentTokens(document, TokenLabel);
  creators["tag-map"] = CreateTFMapFromDocumentTokens(document, TokenTag);
  creators["tag-to-category"] = CreateTagToCategoryFromTokens(document);
  creators["word-map"] = CreateTFMapFromDocumentTokens(document, TokenWord);
  return creators;
}

bool PopulateTestInputs::Populate(
    const std::unordered_map<string, Create> &creator_map,
    TaskContext *context) {
  TaskSpec *spec = context->mutable_spec();
  bool found_all_inputs = true;

  // Fail if a mandatory input is not found.
  auto name_not_found = [&found_all_inputs](TaskInput *input) {
    found_all_inputs = false;
  };

  for (TaskInput &input : *spec->mutable_input()) {
    auto it = creator_map.find(input.name());
    (it == creator_map.end() ? name_not_found : it->second)(&input);

    // Check for compatibility with declared supported formats.
    for (const auto &part : input.part()) {
      if (!TaskContext::Supports(input, part.file_format(),
                                 part.record_format())) {
        LOG(FATAL) << "Input " << input.name()
                   << " does not support file of type " << part.file_format()
                   << "/" << part.record_format();
      }
    }
  }
  return found_all_inputs;
}

PopulateTestInputs::Create PopulateTestInputs::CreateTFMapFromDocumentTokens(
    const Sentence &document,
    std::function<vector<string>(const Token &)> token2str) {
  return [document, token2str](TaskInput *input) {
    TermFrequencyMap map;

    // Build and write the dummy term frequency map.
    for (const Token &token : document.token()) {
      vector<string> strings_for_token = token2str(token);
      for (const string &s : strings_for_token) map.Increment(s);
    }
    string file_name = AddPart(input, "text", "");
    map.Save(file_name);
  };
}

PopulateTestInputs::Create PopulateTestInputs::CreateTagToCategoryFromTokens(
    const Sentence &document) {
  return [document](TaskInput *input) {
    map<string, string> tag_to_category;
    for (auto &token : document.token()) {
      if (token.has_tag()) tag_to_category[token.tag()] = token.category();
    }
    StringToStringMap output_map;
    for (auto &pair : tag_to_category) {
      auto *out_pair = output_map.add_pair();
      out_pair->set_key(pair.first);
      out_pair->set_value(pair.second);
    }
    const string file_name = AddPart(input, "proto", "StringToStringMap");

    tensorflow::WritableFile *file;
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(file_name, &file));
    tensorflow::io::RecordWriter writer(file);
    TF_CHECK_OK(writer.WriteRecord(output_map.SerializeAsString()));
  };
}

vector<string> PopulateTestInputs::TokenCategory(const Token &token) {
  if (token.has_category()) return {token.category()};
  return {};
}

vector<string> PopulateTestInputs::TokenLabel(const Token &token) {
  if (token.has_label()) return {token.label()};
  return {};
}

vector<string> PopulateTestInputs::TokenTag(const Token &token) {
  if (token.has_tag()) return {token.tag()};
  return {};
}

vector<string> PopulateTestInputs::TokenWord(const Token &token) {
  if (token.has_word()) return {token.word()};
  return {};
}

string PopulateTestInputs::AddPart(TaskInput *input, const string &file_format,
                                   const string &record_format) {
  string file_name =
      tensorflow::strings::StrCat(FLAGS_test_tmpdir, "/", input->name());
  auto *part = CHECK_NOTNULL(input)->add_part();
  part->set_file_pattern(file_name);
  part->set_file_format(file_format);
  part->set_record_format(record_format);
  return file_name;
}

}  // namespace neurosis
