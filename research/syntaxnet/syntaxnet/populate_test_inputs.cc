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

#include "syntaxnet/populate_test_inputs.h"

#include <map>
#include <utility>

#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

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
    std::function<std::vector<string>(const Token &)> token2str) {
  return [document, token2str](TaskInput *input) {
    TermFrequencyMap map;

    // Build and write the dummy term frequency map.
    for (const Token &token : document.token()) {
      std::vector<string> strings_for_token = token2str(token);
      for (const string &s : strings_for_token) map.Increment(s);
    }
    string file_name = AddPart(input, "text", "");
    map.Save(file_name);
  };
}

PopulateTestInputs::Create PopulateTestInputs::CreateTagToCategoryFromTokens(
    const Sentence &document) {
  return [document](TaskInput *input) {
    TagToCategoryMap tag_to_category;
    for (auto &token : document.token()) {
      if (token.has_tag()) {
        tag_to_category.SetCategory(token.tag(), token.category());
      }
    }
    const string file_name = AddPart(input, "text", "");
    tag_to_category.Save(file_name);
  };
}

std::vector<string> PopulateTestInputs::TokenCategory(const Token &token) {
  if (token.has_category()) return {token.category()};
  return {};
}

std::vector<string> PopulateTestInputs::TokenLabel(const Token &token) {
  if (token.has_label()) return {token.label()};
  return {};
}

std::vector<string> PopulateTestInputs::TokenTag(const Token &token) {
  if (token.has_tag()) return {token.tag()};
  return {};
}

std::vector<string> PopulateTestInputs::TokenWord(const Token &token) {
  if (token.has_word()) return {token.word()};
  return {};
}

string PopulateTestInputs::AddPart(TaskInput *input, const string &file_format,
                                   const string &record_format) {
  string file_name =
      tensorflow::strings::StrCat(
          tensorflow::testing::TmpDir(), input->name());
  auto *part = CHECK_NOTNULL(input)->add_part();
  part->set_file_pattern(file_name);
  part->set_file_format(file_format);
  part->set_record_format(record_format);
  return file_name;
}

}  // namespace syntaxnet
