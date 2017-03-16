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

#include "syntaxnet/term_frequency_map.h"

#include <stddef.h>
#include <algorithm>
#include <limits>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/regexp.h"

namespace syntaxnet {

int TermFrequencyMap::Increment(const string &term) {
  CHECK_EQ(term_index_.size(), term_data_.size());
  const TermIndex::const_iterator it = term_index_.find(term);
  if (term_index_.find(term) != term_index_.end()) {
    // Increment the existing term.
    std::pair<string, int64> &data = term_data_[it->second];
    CHECK_EQ(term, data.first);
    ++(data.second);
    return it->second;
  } else {
    // Add a new term.
    const int index = term_index_.size();
    CHECK_LT(index, std::numeric_limits<int32>::max());  // overflow
    term_index_[term] = index;
    term_data_.push_back(std::pair<string, int64>(term, 1));
    return index;
  }
}

void TermFrequencyMap::Clear() {
  term_index_.clear();
  term_data_.clear();
}

void TermFrequencyMap::Load(const string &filename, int min_frequency,
                            int max_num_terms) {
  Clear();

  // If max_num_terms is non-positive, replace it with INT_MAX.
  if (max_num_terms <= 0) max_num_terms = std::numeric_limits<int>::max();

  // Read the first line (total # of terms in the mapping).
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
  static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
  tensorflow::io::RandomAccessInputStream stream(file.get());
  tensorflow::io::BufferedInputStream buffer(&stream, kInputBufferSize);
  string line;
  TF_CHECK_OK(buffer.ReadLine(&line));
  int32 total = -1;
  CHECK(utils::ParseInt32(line.c_str(), &total));
  CHECK_GE(total, 0);

  // Read the mapping.
  int64 last_frequency = -1;
  for (int i = 0; i < total && i < max_num_terms; ++i) {
    TF_CHECK_OK(buffer.ReadLine(&line));
    string term;
    int64 frequency = 0;
    CHECK(RE2::FullMatch(line, "(.*) (\\d*)", &term, &frequency));
    CHECK(!term.empty());
    CHECK_GT(frequency, 0);

    // Check frequency sorting (descending order).
    if (i > 0) CHECK_GE(last_frequency, frequency);
    last_frequency = frequency;

    // Ignore low-frequency items.
    if (frequency < min_frequency) continue;

    // Check uniqueness of the mapped terms.
    CHECK(term_index_.find(term) == term_index_.end())
        << "File " << filename << " has duplicate term: " << term;

    // Assign the next available index.
    const int index = term_index_.size();
    term_index_[term] = index;
    term_data_.push_back(std::pair<string, int64>(term, frequency));
  }
  CHECK_EQ(term_index_.size(), term_data_.size());
  LOG(INFO) << "Loaded " << term_index_.size() << " terms from " << filename
            << ".";
}

struct TermFrequencyMap::SortByFrequencyThenTerm {
  // Return a > b to sort in descending order of frequency; otherwise,
  // lexicographic sort on term.
  bool operator()(const std::pair<string, int64> &a,
                  const std::pair<string, int64> &b) const {
    return (a.second > b.second || (a.second == b.second && a.first < b.first));
  }
};

void TermFrequencyMap::Save(const string &filename) const {
  CHECK_EQ(term_index_.size(), term_data_.size());

  // Copy and sort the term data.
  std::vector<std::pair<string, int64>> sorted_data(term_data_);
  std::sort(sorted_data.begin(), sorted_data.end(), SortByFrequencyThenTerm());

  // Write the number of terms.
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
  CHECK_LE(term_index_.size(), std::numeric_limits<int32>::max());  // overflow
  const int32 num_terms = term_index_.size();
  const string header = tensorflow::strings::StrCat(num_terms, "\n");
  TF_CHECK_OK(file->Append(header));

  // Write each term and frequency.
  for (size_t i = 0; i < sorted_data.size(); ++i) {
    if (i > 0) CHECK_GE(sorted_data[i - 1].second, sorted_data[i].second);
    const string line = tensorflow::strings::StrCat(
        sorted_data[i].first, " ", sorted_data[i].second, "\n");
    TF_CHECK_OK(file->Append(line));
  }
  TF_CHECK_OK(file->Close()) << "for file " << filename;
  LOG(INFO) << "Saved " << term_index_.size() << " terms to " << filename
            << ".";
}

TagToCategoryMap::TagToCategoryMap(const string &filename) {
  // Load the mapping.
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(filename, &file));
  static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
  tensorflow::io::RandomAccessInputStream stream(file.get());
  tensorflow::io::BufferedInputStream buffer(&stream, kInputBufferSize);
  string line;
  while (buffer.ReadLine(&line) == tensorflow::Status::OK()) {
    std::vector<string> pair = utils::Split(line, '\t');
    CHECK(line.empty() || pair.size() == 2) << line;
    tag_to_category_[pair[0]] = pair[1];
  }
}

// Returns the category associated with the given tag.
const string &TagToCategoryMap::GetCategory(const string &tag) const {
  const auto it = tag_to_category_.find(tag);
  CHECK(it != tag_to_category_.end()) << "No category found for tag " << tag;
  return it->second;
}

void TagToCategoryMap::SetCategory(const string &tag, const string &category) {
  const auto it = tag_to_category_.find(tag);
  if (it != tag_to_category_.end()) {
    if (category != it->second) {
      invalid_mappings_[tag].insert(it->second);
      invalid_mappings_[tag].insert(category);
    }
  } else {
    tag_to_category_[tag] = category;
  }
}

void TagToCategoryMap::Save(const string &filename) const {
  for (auto &pair : invalid_mappings_) {
    LOG(ERROR)
        << "Warning: POS tag is being mapped to multiple coarse POS tags. "
        << "'" << pair.first << "' is mapped to " << pair.second.size()
        << " categories:";
    for (auto &category : pair.second) {
      LOG(ERROR) << category;
    }
    LOG(ERROR) << "Recommend setting "
               << "join_category_to_pos to 'true' in this case.";
  }

  // Write tag and category on each line.
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));
  for (const auto &pair : tag_to_category_) {
    const string line =
        tensorflow::strings::StrCat(pair.first, "\t", pair.second, "\n");
    TF_CHECK_OK(file->Append(line));
  }
  TF_CHECK_OK(file->Close()) << "for file " << filename;
}

}  // namespace syntaxnet
