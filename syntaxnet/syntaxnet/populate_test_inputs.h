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

// A utility for populating a set of inputs of a task.  This knows how to create
// tag-map, category-map, label-map and has hooks to
// populate other kinds of inputs.  The expected set of operations are:
//
// Sentence document_for_init = ...;
// TaskContext context;
// context->SetParameter("my_parameter", "true");
// MyDocumentProcessor processor;
// processor.Setup(&context);
// PopulateTestInputs::Defaults(document_for_init).Populate(&context);
// processor.Init(&context);
//
// This will check the inputs requested by the processor's Setup(TaskContext *)
// function, and files corresponding to them.  For example, if the processor
// asked for the a "tag-map" input, it will create a TermFrequencyMap, populate
// it with the POS tags found in the Sentence document_for_init, save it to disk
// and update the TaskContext with the location of the file.  By convention, the
// location is the name of the input. Conceptually, the logic is very simple:
//
// for (TaskInput &input : context->mutable_spec()->mutable_input()) {
//   creators[input.name()](&input);
//   // check for missing inputs, incompatible formats, etc...
// }
//
// The Populate() routine will also check compatability between requested and
// supplied formats. The Default mapping knows how to populate the following
// inputs:
//
//  - category-map: TermFrequencyMap containing POS categories.
//
//  - label-map: TermFrequencyMap containing parser labels.
//
//  - tag-map: TermFrequencyMap containing POS tags.
//
//  - tag-to-category: StringToStringMap mapping POS tags to categories.
//
//  - word-map: TermFrequencyMap containing words.
//
// Clients can add creation routines by defining a std::function:
//
// auto creators = PopulateTestInputs::Defaults(document_for_init);
// creators["my-input"] = [](TaskInput *input) { ...; }
//
// See also creators.Add() for more convenience functions.

#ifndef SYNTAXNET_POPULATE_TEST_INPUTS_H_
#define SYNTAXNET_POPULATE_TEST_INPUTS_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "syntaxnet/utils.h"

namespace syntaxnet {

class Sentence;
class TaskContext;
class TaskInput;
class TaskOutput;
class Token;

class PopulateTestInputs {
 public:
  // When called, Create() should populate an input by creating a file and
  // adding one or more parts to the TaskInput.
  typedef std::function<void(TaskInput *)> Create;

  // When called, CreateFile() should create a file resource at the given
  // path. These are typically less inconvient to write.
  typedef std::function<void(const string &)> CreateFile;

  // A set of creators, one for each input in a TaskContext.
  class CreatorMap : public std::unordered_map<string, Create> {
   public:
    // A simplified way to add a single-file creator.  The name of the file
    // location will be file::JoinPath(FLAGS_test_tmpdir, name).
    void Add(const string &name, const string &file_format,
             const string &record_format, CreateFile makefile);

    // Convenience method to populate the inputs in context.  Returns true if it
    // was possible to populate each input, and false otherwise.  If a mandatory
    // input does not have a creator, then we LOG(FATAL).
    bool Populate(TaskContext *context) const;
  };

  // Default creator set.  This knows how to generate from a given Document
  //  - category-map
  //  - label-map
  //  - tag-map
  //  - tag-to-category
  //  - word-map
  //
  //  Note: the default creators capture the document input by value: this means
  //  that subsequent modifications to the document will NOT be
  //  reflected in the inputs. However, the following is perfectly valid:
  //
  //  CreatorMap creators;
  //  {
  //    Sentence document;
  //    creators = PopulateTestInputs::Defaults(document);
  //  }
  //  creators.Populate(context);
  static CreatorMap Defaults(const Sentence &document);

  // Populates the TaskContext object from a map of creator functions. Note that
  // this static version is compatible with any hash map of the correct type.
  static bool Populate(const std::unordered_map<string, Create> &creator_map,
                       TaskContext *context);

  // Helper function for creating a term frequency map from a document.  This
  // iterates over all the tokens in the document, calls token2str on each
  // token, and adds each returned string to the term frequency map.  The map is
  // then saved to FLAGS_test_tmpdir/name.
  static Create CreateTFMapFromDocumentTokens(
      const Sentence &document,
      std::function<vector<string>(const Token &)> token2str);

  // Creates a StringToStringMap protocol buffer input that maps tags to
  // categories. Uses whatever mapping is present in the document.
  static Create CreateTagToCategoryFromTokens(const Sentence &document);

  // Default implementations for "token2str" above.
  static vector<string> TokenCategory(const Token &token);
  static vector<string> TokenLabel(const Token &token);
  static vector<string> TokenTag(const Token &token);
  static vector<string> TokenWord(const Token &token);

  // Utility function. Sets the TaskInput->part() fields for a new input part.
  // Returns the file name.
  static string AddPart(TaskInput *input, const string &file_format,
                        const string &record_format);
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_POPULATE_TEST_INPUTS_H_
