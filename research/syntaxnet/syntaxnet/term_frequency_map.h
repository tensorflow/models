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

#ifndef SYNTAXNET_TERM_FREQUENCY_MAP_H_
#define SYNTAXNET_TERM_FREQUENCY_MAP_H_

#include <stddef.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/core/status.h"


namespace syntaxnet {

// A mapping from strings to frequencies with save and load functionality.
class TermFrequencyMap {
 public:
  // Creates an empty frequency map.
  TermFrequencyMap() {}

  // Creates a term frequency map by calling Load.
  TermFrequencyMap(const string &file, int min_frequency, int max_num_terms) {
    Load(file, min_frequency, max_num_terms);
  }

  // Returns the number of terms with positive frequency.
  int Size() const { return term_index_.size(); }

  // Returns the index associated with the given term.  If the term does not
  // exist, the unknown index is returned instead.
  int LookupIndex(const string &term, int unknown) const {
    const TermIndex::const_iterator it = term_index_.find(term);
    return (it != term_index_.end() ? it->second : unknown);
  }

  // Returns the term associated with the given index.
  const string &GetTerm(int index) const { return term_data_[index].first; }

  // Returns the frequency associated with the given index.
  int64 GetFrequency(int index) const { return term_data_[index].second; }

  // Increases the frequency of the given term by 1, creating a new entry if
  // necessary, and returns the index of the term.
  int Increment(const string &term);

  // Clears all frequencies.
  void Clear();

  // Loads a frequency mapping from the given file, which must have been created
  // by an earlier call to Save().  On error, returns non-OK.
  //
  // After loading, the term indices are guaranteed to be ordered in descending
  // order of frequency (breaking ties arbitrarily).  However, any new terms
  // inserted after loading do not maintain this sorting invariant.
  //
  // Only loads terms with frequency >= min_frequency.  If max_num_terms <= 0,
  // then all qualifying terms are loaded; otherwise, max_num_terms terms with
  // maximal frequency are loaded (breaking ties arbitrarily).
  tensorflow::Status TryLoad(const string &filename, int min_frequency,
                             int max_num_terms);

  // Like TryLoad(), but fails on error.
  void Load(const string &filename, int min_frequency, int max_num_terms);

  // Saves a frequency mapping to the given file.
  void Save(const string &filename) const;

 private:
  // Hashtable for term-to-index mapping.
  using TermIndex =  std::unordered_map<string, int>;


  // Sorting functor for term data.
  struct SortByFrequencyThenTerm;

  // Mapping from terms to indices.
  TermIndex term_index_;

  // Mapping from indices to term and frequency.
  std::vector<std::pair<string, int64>> term_data_;

  TF_DISALLOW_COPY_AND_ASSIGN(TermFrequencyMap);
};

// A mapping from tags to categories.
class TagToCategoryMap {
 public:
  TagToCategoryMap() {}
  ~TagToCategoryMap() {}

  // Loads a tag to category map from a text file.
  explicit TagToCategoryMap(const string &filename);

  // Sets the category for the given tag.
  void SetCategory(const string &tag, const string &category);

  // Returns the category associated with the given tag.
  const string &GetCategory(const string &tag) const;

  // Saves a tag to category map to the given file.
  void Save(const string &filename) const;

 private:
  // List of tags that have multiple coarse tags, and their mappings. Used only
  // for error reporting at Save() time.
  std::map<string, std::unordered_set<string>> invalid_mappings_;

  std::map<string, string> tag_to_category_;

  TF_DISALLOW_COPY_AND_ASSIGN(TagToCategoryMap);
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_TERM_FREQUENCY_MAP_H_
