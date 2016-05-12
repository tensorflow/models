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

#ifndef $TARGETDIR_AFFIX_H_
#define $TARGETDIR_AFFIX_H_

#include <stddef.h>
#include <string>
#include <vector>

#include "syntaxnet/utils.h"
#include "syntaxnet/dictionary.pb.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/proto_io.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/workspace.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

// An affix represents a prefix or suffix of a word of a certain length. Each
// affix has a unique id and a textual form. An affix also has a pointer to the
// affix that is one character shorter. This creates a chain of affixes that are
// successively shorter.
class Affix {
 private:
  friend class AffixTable;
  Affix(int id, const char *form, int length)
      : id_(id), length_(length), form_(form), shorter_(NULL), next_(NULL) {}

 public:
  // Returns unique id of affix.
  int id() const { return id_; }

  // Returns the textual representation of the affix.
  string form() const { return form_; }

  // Returns the length of the affix.
  int length() const { return length_; }

  // Gets/sets the affix that is one character shorter.
  Affix *shorter() const { return shorter_; }
  void set_shorter(Affix *next) { shorter_ = next; }

 private:
  // Affix id.
  int id_;

  // Length (in characters) of affix.
  int length_;

  // Text form of affix.
  string form_;

  // Pointer to affix that is one character shorter.
  Affix *shorter_;

  // Next affix in bucket chain.
  Affix *next_;

  TF_DISALLOW_COPY_AND_ASSIGN(Affix);
};

// An affix table holds all prefixes/suffixes of all the words added to the
// table up to a maximum length. The affixes are chained together to enable
// fast lookup of all affixes for a word.
class AffixTable {
 public:
  // Affix table type.
  enum Type { PREFIX, SUFFIX };

  AffixTable(Type type, int max_length);
  ~AffixTable();

  // Resets the affix table and initialize the table for affixes of up to the
  // maximum length specified.
  void Reset(int max_length);

  // De-serializes this from the given proto.
  void Read(const AffixTableEntry &table_entry);

  // De-serializes this from the given records.
  void Read(ProtoRecordReader *reader);

  // Serializes this to the given proto.
  void Write(AffixTableEntry *table_entry) const;

  // Serializes this to the given records.
  void Write(ProtoRecordWriter *writer) const;

  // Adds all prefixes/suffixes of the word up to the maximum length to the
  // table. The longest affix is returned. The pointers in the affix can be
  // used for getting shorter affixes.
  Affix *AddAffixesForWord(const char *word, size_t size);

  // Gets the affix information for the affix with a certain id. Returns NULL if
  // there is no affix in the table with this id.
  Affix *GetAffix(int id) const;

  // Gets affix form from id. If the affix does not exist in the table, an empty
  // string is returned.
  string AffixForm(int id) const;

  // Gets affix id for affix. If the affix does not exist in the table, -1 is
  // returned.
  int AffixId(const string &form) const;

  // Returns size of the affix table.
  int size() const { return affixes_.size(); }

  // Returns the maximum affix length.
  int max_length() const { return max_length_; }

 private:
  // Adds a new affix to table.
  Affix *AddNewAffix(const string &form, int length);

  // Finds existing affix in table.
  Affix *FindAffix(const string &form) const;

  // Resizes bucket array.
  void Resize(int size_hint);

  // Affix type (prefix or suffix).
  Type type_;

  // Maximum length of affix.
  int max_length_;

  // Index from affix ids to affix items.
  vector<Affix *> affixes_;

  // Buckets for word-to-affix hash map.
  vector<Affix *> buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(AffixTable);
};

}  // namespace syntaxnet

#endif  // $TARGETDIR_AFFIX_H_
