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

#include "syntaxnet/affix.h"

#include <ctype.h>
#include <string.h>
#include <functional>
#include <string>

#include "syntaxnet/shared_store.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "syntaxnet/workspace.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"
#include "util/utf8/unicodetext.h"

namespace syntaxnet {

// Initial number of buckets in term and affix hash maps. This must be a power
// of two.
static const int kInitialBuckets = 1024;

// Fill factor for term and affix hash maps.
static const int kFillFactor = 2;

int TermHash(const string &term) {
  return utils::Hash32(term.data(), term.size(), 0xDECAF);
}

// Copies a substring of a Unicode text to a string.
static void UnicodeSubstring(const UnicodeText::const_iterator &start,
                             const UnicodeText::const_iterator &end,
                             string *result) {
  result->clear();
  result->append(start.utf8_data(), end.utf8_data() - start.utf8_data());
}

AffixTable::AffixTable(Type type, int max_length) {
  type_ = type;
  max_length_ = max_length;
  Resize(0);
}

AffixTable::~AffixTable() { Reset(0); }

void AffixTable::Reset(int max_length) {
  // Save new maximum affix length.
  max_length_ = max_length;

  // Delete all data.
  for (size_t i = 0; i < affixes_.size(); ++i) delete affixes_[i];
  affixes_.clear();
  buckets_.clear();
  Resize(0);
}

void AffixTable::Read(const AffixTableEntry &table_entry) {
  CHECK_EQ(table_entry.type(), type_ == PREFIX ? "PREFIX" : "SUFFIX");
  CHECK_GE(table_entry.max_length(), 0);
  Reset(table_entry.max_length());

  // First, create all affixes.
  for (int affix_id = 0; affix_id < table_entry.affix_size(); ++affix_id) {
    const auto &affix_entry = table_entry.affix(affix_id);
    CHECK_GE(affix_entry.length(), 0);
    CHECK_LE(affix_entry.length(), max_length_);
    CHECK(FindAffix(affix_entry.form()) == nullptr);  // forbid duplicates
    Affix *affix = AddNewAffix(affix_entry.form(), affix_entry.length());
    CHECK_EQ(affix->id(), affix_id);
  }
  CHECK_EQ(affixes_.size(), table_entry.affix_size());

  // Next, link the shorter affixes.
  for (int affix_id = 0; affix_id < table_entry.affix_size(); ++affix_id) {
    const auto &affix_entry = table_entry.affix(affix_id);
    if (affix_entry.shorter_id() == -1) {
      CHECK_EQ(affix_entry.length(), 1);
      continue;
    }
    CHECK_GT(affix_entry.length(), 1);
    CHECK_GE(affix_entry.shorter_id(), 0);
    CHECK_LT(affix_entry.shorter_id(), affixes_.size());
    Affix *affix = affixes_[affix_id];
    Affix *shorter = affixes_[affix_entry.shorter_id()];
    CHECK_EQ(affix->length(), shorter->length() + 1);
    affix->set_shorter(shorter);
  }
}

void AffixTable::Read(ProtoRecordReader *reader) {
  AffixTableEntry table_entry;
  TF_CHECK_OK(reader->Read(&table_entry));
  Read(table_entry);
}

void AffixTable::Write(AffixTableEntry *table_entry) const {
  table_entry->Clear();
  table_entry->set_type(type_ == PREFIX ? "PREFIX" : "SUFFIX");
  table_entry->set_max_length(max_length_);
  for (const Affix *affix : affixes_) {
    auto *affix_entry = table_entry->add_affix();
    affix_entry->set_form(affix->form());
    affix_entry->set_length(affix->length());
    affix_entry->set_shorter_id(
        affix->shorter() == nullptr ? -1 : affix->shorter()->id());
  }
}

void AffixTable::Write(ProtoRecordWriter *writer) const {
  AffixTableEntry table_entry;
  Write(&table_entry);
  writer->Write(table_entry);
}

Affix *AffixTable::AddAffixesForWord(const char *word, size_t size) {
  // The affix length is measured in characters and not bytes so we need to
  // determine the length in characters.
  UnicodeText text;
  text.PointToUTF8(word, size);
  int length = text.size();

  // Determine longest affix.
  int affix_len = length;
  if (affix_len > max_length_) affix_len = max_length_;
  if (affix_len == 0) return nullptr;

  // Find start and end of longest affix.
  UnicodeText::const_iterator start, end;
  if (type_ == PREFIX) {
    start = end = text.begin();
    for (int i = 0; i < affix_len; ++i) ++end;
  } else {
    start = end = text.end();
    for (int i = 0; i < affix_len; ++i) --start;
  }

  // Try to find successively shorter affixes.
  Affix *top = nullptr;
  Affix *ancestor = nullptr;
  string s;
  while (affix_len > 0) {
    // Try to find affix in table.
    UnicodeSubstring(start, end, &s);
    Affix *affix = FindAffix(s);
    if (affix == nullptr) {
      // Affix not found, add new one to table.
      affix = AddNewAffix(s, affix_len);

      // Update ancestor chain.
      if (ancestor != nullptr) ancestor->set_shorter(affix);
      ancestor = affix;
      if (top == nullptr) top = affix;
    } else {
      // Affix found. Update ancestor if needed and return match.
      if (ancestor != nullptr) ancestor->set_shorter(affix);
      if (top == nullptr) top = affix;
      break;
    }

    // Next affix.
    if (type_ == PREFIX) {
      --end;
    } else {
      ++start;
    }

    affix_len--;
  }

  return top;
}

Affix *AffixTable::GetAffix(int id) const {
  if (id < 0 || id >= static_cast<int>(affixes_.size())) {
    return nullptr;
  } else {
    return affixes_[id];
  }
}

string AffixTable::AffixForm(int id) const {
  Affix *affix = GetAffix(id);
  if (affix == nullptr) {
    return "";
  } else {
    return affix->form();
  }
}

int AffixTable::AffixId(const string &form) const {
  Affix *affix = FindAffix(form);
  if (affix == nullptr) {
    return -1;
  } else {
    return affix->id();
  }
}

Affix *AffixTable::AddNewAffix(const string &form, int length) {
  int hash = TermHash(form);
  int id = affixes_.size();
  if (id > static_cast<int>(buckets_.size()) * kFillFactor) Resize(id);
  int b = hash & (buckets_.size() - 1);

  // Create new affix object.
  Affix *affix = new Affix(id, form.c_str(), length);
  affixes_.push_back(affix);

  // Insert affix in bucket chain.
  affix->next_ = buckets_[b];
  buckets_[b] = affix;

  return affix;
}

Affix *AffixTable::FindAffix(const string &form) const {
  // Compute hash value for word.
  int hash = TermHash(form);

  // Try to find affix in hash table.
  Affix *affix = buckets_[hash & (buckets_.size() - 1)];
  while (affix != nullptr) {
    if (strcmp(affix->form_.c_str(), form.c_str()) == 0) return affix;
    affix = affix->next_;
  }
  return nullptr;
}

void AffixTable::Resize(int size_hint) {
  // Compute new size for bucket array.
  int new_size = kInitialBuckets;
  while (new_size < size_hint) new_size *= 2;
  int mask = new_size - 1;

  // Distribute affixes in new buckets.
  buckets_.resize(new_size);
  for (size_t i = 0; i < buckets_.size(); ++i) {
    buckets_[i] = nullptr;
  }
  for (size_t i = 0; i < affixes_.size(); ++i) {
    Affix *affix = affixes_[i];
    int b = TermHash(affix->form_) & mask;
    affix->next_ = buckets_[b];
    buckets_[b] = affix;
  }
}

}  // namespace syntaxnet
