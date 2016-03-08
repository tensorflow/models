/* -*- Mode: C++ -*- */

/*
 * Copyright 2016 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This program starts with a text file (and optionally a vocabulary file) and
 * computes co-occurrence statistics. It emits output in a format that can be
 * consumed by the "swivel" program.  It's functionally equivalent to "prep.py",
 * but works much more quickly.
 */

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

static const char usage[] = R"(
Prepares a corpus for processing by Swivel.

Usage:

  prep --output_dir <output-dir> --input <text-file>

Options:

  --input <filename>
      The input text.

  --output_dir <directory>
      Specifies the output directory where the various Swivel data
      files should be placed.  This directory must exist.

  --shard_size <int>
      Specifies the shard size; default 4096.

  --min_count <int>
      The minimum number of times a word should appear to be included in the
      generated vocabulary; default 5.  (Ignored if --vocab is used.)

  --max_vocab <int>
      The maximum vocabulary size to generate from the input corpus; default
      102,400.  (Ignored if --vocab is used.)

  --vocab <filename>
      Use the specified unigram vocabulary instead of generating
      it from the corpus.

  --window_size <int>
      Specifies the window size for computing co-occurrence stats;
      default 10.
)";

struct cooc_t {
  int row;
  int col;
  float cnt;
};

typedef std::map<long long, float> cooc_counts_t;

// Retrieves the next word from the input stream, treating words as simply being
// delimited by whitespace.  Returns true if this is the end of a "sentence";
// i.e., a newline.
bool NextWord(std::ifstream &fin, std::string* word) {
  std::string buf;
  char c;

  if (fin.eof()) {
    word->erase();
    return true;
  }

  // Skip leading whitespace.
  do {
    c = fin.get();
  } while (!fin.eof() && std::isspace(c));

  if (fin.eof()) {
    word->erase();
    return true;
  }

  // Read the next word.
  do {
    buf += c;
    c = fin.get();
  } while (!fin.eof() && !std::isspace(c));

  *word = buf;
  if (c == '\n' || fin.eof()) return true;

  // Skip trailing whitespace.
  do {
    c = fin.get();
  } while (!fin.eof() && std::isspace(c));

  if (fin.eof()) return true;

  fin.unget();
  return false;
}

// Creates a vocabulary from the most frequent terms in the input file.
std::vector<std::string> CreateVocabulary(const std::string input_filename,
                                          const int shard_size,
                                          const int min_vocab_count,
                                          const int max_vocab_size) {
  std::vector<std::string> vocab;

  // Count all the distinct tokens in the file.  (XXX this will eventually
  // consume all memory and should be re-written to periodically trim the data.)
  std::unordered_map<std::string, long long> counts;

  std::ifstream fin(input_filename, std::ifstream::ate);

  if (!fin) {
    std::cerr << "couldn't read input file '" << input_filename << "'"
              << std::endl;

    return vocab;
  }

  const auto input_size = fin.tellg();
  fin.seekg(0);

  long long ntokens = 0;
  while (!fin.eof()) {
    std::string word;
    NextWord(fin, &word);
    counts[word] += 1;

    if (++ntokens % 1000000 == 0) {
      const float pct = 100.0 * static_cast<float>(fin.tellg()) / input_size;
      fprintf(stdout, "\rComputing vocabulary: %0.1f%% complete...", pct);
      std::flush(std::cout);
    }
  }

  std::cout << counts.size() << " distinct tokens" << std::endl;

  // Sort the vocabulary from most frequent to least frequent.
  std::vector<std::pair<std::string, long long>> buf;
  std::copy(counts.begin(), counts.end(), std::back_inserter(buf));
  std::sort(buf.begin(), buf.end(),
            [](const std::pair<std::string, long long> &a,
               const std::pair<std::string, long long> &b) {
              return b.second < a.second;
            });

  // Truncate to the maximum vocabulary size
  if (static_cast<int>(buf.size()) > max_vocab_size) buf.resize(max_vocab_size);
  if (buf.empty()) return vocab;

  // Eliminate rare tokens and truncate to a size modulo the shard size.
  int vocab_size = buf.size();
  while (vocab_size > 0 && buf[vocab_size - 1].second < min_vocab_count)
    --vocab_size;

  vocab_size -= vocab_size % shard_size;
  if (static_cast<int>(buf.size()) > vocab_size) buf.resize(vocab_size);

  // Copy out the tokens.
  for (const auto& pair : buf) vocab.push_back(pair.first);

  return vocab;
}

std::vector<std::string> ReadVocabulary(const std::string vocab_filename) {
  std::vector<std::string> vocab;

  std::ifstream fin(vocab_filename);
  int index = 0;
  for (std::string token; std::getline(fin, token); ++index) {
    auto n = token.find('\t');
    if (n != std::string::npos) token = token.substr(n);

    vocab.push_back(token);
  }

  return vocab;
}

void WriteVocabulary(const std::vector<std::string> &vocab,
                     const std::string &output_dirname) {
  for (const std::string filename : {"row_vocab.txt", "col_vocab.txt"}) {
    std::ofstream fout(output_dirname + "/" + filename);
    for (const auto &token : vocab) fout << token << std::endl;
  }
}

// Manages accumulation of co-occurrence data into temporary disk buffer files.
class CoocBuffer {
 public:
  CoocBuffer(const std::string &output_dirname, const int num_shards,
             const int shard_size);

  // Accumulate the co-occurrence counts to the buffer.
  void AccumulateCoocs(const cooc_counts_t &coocs);

  // Read the buffer to produce shard files.
  void WriteShards();

 protected:
  // The output directory. Also used for temporary buffer files.
  const std::string output_dirname_;

  // The number of row/column shards.
  const int num_shards_;

  // The number of elements per shard.
  const int shard_size_;

  // Parallel arrays of temporary file paths and file descriptors.
  std::vector<std::string> paths_;
  std::vector<int> fds_;

  // Ensures that only one buffer file is getting written at a time.
  pthread_mutex_t writer_mutex_;
};

CoocBuffer::CoocBuffer(const std::string &output_dirname, const int num_shards,
                       const int shard_size)
    : output_dirname_(output_dirname),
      num_shards_(num_shards),
      shard_size_(shard_size),
      writer_mutex_(PTHREAD_MUTEX_INITIALIZER) {
  for (int row = 0; row < num_shards_; ++row) {
    for (int col = 0; col < num_shards_; ++col) {
      char filename[256];
      sprintf(filename, "shard-%03d-%03d.tmp", row, col);

      std::string path = output_dirname + "/" + filename;
      int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
      assert(fd > 0);

      paths_.push_back(path);
      fds_.push_back(fd);
    }
  }
}

void CoocBuffer::AccumulateCoocs(const cooc_counts_t &coocs) {
  std::vector<std::vector<cooc_t>> bufs(fds_.size());

  for (const auto &cooc : coocs) {
    const int row_id = cooc.first >> 32;
    const int col_id = cooc.first & 0xffffffff;
    const float cnt = cooc.second;

    const int row_shard = row_id % num_shards_;
    const int row_off = row_id / num_shards_;
    const int col_shard = col_id % num_shards_;
    const int col_off = col_id / num_shards_;

    const int top_shard_idx = row_shard * num_shards_ + col_shard;
    bufs[top_shard_idx].push_back(cooc_t{row_off, col_off, cnt});

    const int bot_shard_idx = col_shard * num_shards_ + row_shard;
    bufs[bot_shard_idx].push_back(cooc_t{col_off, row_off, cnt});
  }

  // XXX TODO: lock
  for (int i = 0; i < static_cast<int>(fds_.size()); ++i) {
    int rv = pthread_mutex_lock(&writer_mutex_);
    assert(rv == 0);
    const int nbytes = bufs[i].size() * sizeof(cooc_t);
    int nwritten = write(fds_[i], bufs[i].data(), nbytes);
    assert(nwritten == nbytes);
    pthread_mutex_unlock(&writer_mutex_);
  }
}

void CoocBuffer::WriteShards() {
  for (int shard = 0; shard < static_cast<int>(fds_.size()); ++shard) {
    const int row_shard = shard / num_shards_;
    const int col_shard = shard % num_shards_;

    std::cout << "\rwriting shard " << (shard + 1) << "/"
              << (num_shards_ * num_shards_);
    std::flush(std::cout);

    // Construct the tf::Example proto.  First, we add the global rows and
    // column that are present in the shard.
    tensorflow::Example example;

    auto &feature = *example.mutable_features()->mutable_feature();
    auto global_row = feature["global_row"].mutable_int64_list();
    auto global_col = feature["global_col"].mutable_int64_list();

    for (int i = 0; i < shard_size_; ++i) {
      global_row->add_value(row_shard + i * num_shards_);
      global_col->add_value(col_shard + i * num_shards_);
    }

    // Next we add co-occurrences as a sparse representation.  Map the
    // co-occurrence counts that we've spooled off to disk: these are in
    // arbitrary order and may contain duplicates.
    const off_t nbytes = lseek(fds_[shard], 0, SEEK_END);
    cooc_t *coocs = static_cast<cooc_t*>(
        mmap(0, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fds_[shard], 0));

    const int ncoocs = nbytes / sizeof(cooc_t);
    cooc_t* cur = coocs;
    cooc_t* end = coocs + ncoocs;

    auto sparse_value = feature["sparse_value"].mutable_float_list();
    auto sparse_local_row = feature["sparse_local_row"].mutable_int64_list();
    auto sparse_local_col = feature["sparse_local_col"].mutable_int64_list();

    std::sort(cur, end, [](const cooc_t &a, const cooc_t &b) {
      return a.row < b.row || (a.row == b.row && a.col < b.col);
    });

    // Accumulate the counts into the protocol buffer.
    int last_row = -1, last_col = -1;
    float count = 0;
    for (; cur != end; ++cur) {
      if (cur->row != last_row || cur->col != last_col) {
        if (last_row >= 0 && last_col >= 0) {
          sparse_local_row->add_value(last_row);
          sparse_local_col->add_value(last_col);
          sparse_value->add_value(count);
        }

        last_row = cur->row;
        last_col = cur->col;
        count = 0;
      }

      count += cur->cnt;
    }

    if (last_row >= 0 && last_col >= 0) {
      sparse_local_row->add_value(last_row);
      sparse_local_col->add_value(last_col);
      sparse_value->add_value(count);
    }

    munmap(coocs, nbytes);
    close(fds_[shard]);

    // Write the protocol buffer as a binary blob to disk.
    char filename[256];
    snprintf(filename, sizeof(filename), "shard-%03d-%03d.pb", row_shard,
             col_shard);

    const std::string path = output_dirname_ + "/" + filename;
    int fd = open(path.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
    assert(fd != -1);

    google::protobuf::io::FileOutputStream fout(fd);
    example.SerializeToZeroCopyStream(&fout);
    fout.Close();

    // Remove the temporary file.
    unlink(paths_[shard].c_str());
  }

  std::cout << std::endl;
}

// Counts the co-occurrences in part of the file.
class CoocCounter {
 public:
  CoocCounter(const std::string &input_filename, const off_t start,
              const off_t end, const int window_size,
              const std::unordered_map<std::string, int> &token_to_id_map,
              CoocBuffer *coocbuf)
      : fin_(input_filename, std::ifstream::ate),
        start_(start),
        end_(end),
        window_size_(window_size),
        token_to_id_map_(token_to_id_map),
        coocbuf_(coocbuf),
        marginals_(token_to_id_map.size()) {}

  // PTthreads-friendly thunk to Count.
  static void* Run(void* param) {
    CoocCounter* self = static_cast<CoocCounter*>(param);
    self->Count();
    return nullptr;
  }

  // Counts the co-occurrences.
  void Count();

  const std::vector<double>& Marginals() const { return marginals_; }

 protected:
  // The input stream.
  std::ifstream fin_;

  // The range of the file to which this counter should attend.
  const off_t start_;
  const off_t end_;

  // The window size for computing co-occurrences.
  const int window_size_;

  // A reference to the mapping from tokens to IDs.
  const std::unordered_map<std::string, int> &token_to_id_map_;

  // The buffer into which counts are to be accumulated.
  CoocBuffer* coocbuf_;

  // The marginal counts accumulated by this counter.
  std::vector<double> marginals_;
};

void CoocCounter::Count() {
  const int max_coocs_size = 16 * 1024 * 1024;

  // A buffer of co-occurrence counts that we'll periodically sort into
  // shards.
  cooc_counts_t coocs;

  fin_.seekg(start_);

  int nlines = 0;
  for (off_t filepos = start_; filepos < end_; filepos = fin_.tellg()) {
    // Buffer a single sentence.
    std::vector<int> sentence;
    bool eos;
    do {
      std::string word;
      eos = NextWord(fin_, &word);
      auto it = token_to_id_map_.find(word);
      if (it != token_to_id_map_.end()) sentence.push_back(it->second);
    } while (!eos);

    // Generate the co-occurrences for the sentence.
    for (int pos = 0; pos < static_cast<int>(sentence.size()); ++pos) {
      const int left_id = sentence[pos];

      const int window_extent =
          std::min(static_cast<int>(sentence.size()) - pos, 1 + window_size_);

      for (int off = 1; off < window_extent; ++off) {
        const int right_id = sentence[pos + off];
        const double count = 1.0 / static_cast<double>(off);
        const long long lo = std::min(left_id, right_id);
        const long long hi = std::max(left_id, right_id);
        const long long key = (hi << 32) | lo;
        coocs[key] += count;

        marginals_[left_id] += count;
        marginals_[right_id] += count;
      }

      marginals_[left_id] += 1.0;
      const long long key = (static_cast<long long>(left_id) << 32) |
                            static_cast<long long>(left_id);

      coocs[key] += 0.5;
    }

    // Periodically flush the co-occurrences to disk.
    if (coocs.size() > max_coocs_size) {
      coocbuf_->AccumulateCoocs(coocs);
      coocs.clear();
    }

    if (start_ == 0 && ++nlines % 1000 == 0) {
      const double pct = 100.0 * filepos / end_;
      fprintf(stdout, "\rComputing co-occurrences: %0.1f%% complete...", pct);
      std::flush(std::cout);
    }
  }

  // Accumulate anything we haven't flushed yet.
  coocbuf_->AccumulateCoocs(coocs);

  if (start_ == 0) std::cout << "done." << std::endl;
}

void WriteMarginals(const std::vector<double> &marginals,
                    const std::string &output_dirname) {
  for (const std::string filename : {"row_sums.txt", "col_sums.txt"}) {
    std::ofstream fout(output_dirname + "/" + filename);
    fout.setf(std::ios::fixed);
    for (double sum : marginals) fout << sum << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::string input_filename;
  std::string vocab_filename;
  std::string output_dirname;
  bool generate_vocab = true;
  int max_vocab_size = 100 * 1024;
  int min_vocab_count = 5;
  int window_size = 10;
  int shard_size = 4096;
  int num_threads = 4;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--vocab") {
      if (++i >= argc) goto argmissing;
      generate_vocab = false;
      vocab_filename = argv[i];
    } else if (arg == "--max_vocab") {
      if (++i >= argc) goto argmissing;
      if ((max_vocab_size = atoi(argv[i])) <= 0) goto badarg;
    } else if (arg == "--min_count") {
      if (++i >= argc) goto argmissing;
      if ((min_vocab_count = atoi(argv[i])) <= 0) goto badarg;
    } else if (arg == "--window_size") {
      if (++i >= argc) goto argmissing;
      if ((window_size = atoi(argv[i])) <= 0) goto badarg;
    } else if (arg == "--input") {
      if (++i >= argc) goto argmissing;
      input_filename = argv[i];
    } else if (arg == "--output_dir") {
      if (++i >= argc) goto argmissing;
      output_dirname = argv[i];
    } else if (arg == "--shard_size") {
      if (++i >= argc) goto argmissing;
      shard_size = atoi(argv[i]);
    } else if (arg == "--num_threads") {
      if (++i >= argc) goto argmissing;
      num_threads = atoi(argv[i]);
    } else if (arg == "--help") {
      std::cout << usage << std::endl;
      return 0;
    } else {
      std::cerr << "unknown arg '" << arg << "'; try --help?" << std::endl;
      return 2;
    }

    continue;

  badarg:
    std::cerr << "'" << argv[i] << "' is not a valid value for '" << arg
              << "'; try --help?" << std::endl;

    return 2;

  argmissing:
    std::cerr << arg << " requires an argument; try --help?" << std::endl;
  }

  if (input_filename.empty()) {
    std::cerr << "please specify the input text with '--input'; try --help?"
              << std::endl;
    return 2;
  }

  if (output_dirname.empty()) {
    std::cerr << "please specify the output directory with '--output_dir'"
              << std::endl;

    return 2;
  }

  struct stat sb;
  if (lstat(output_dirname.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
    std::cerr << "output directory '" << output_dirname
              << "' does not exist of is not a directory." << std::endl;

    return 1;
  }

  if (lstat(input_filename.c_str(), &sb) != 0 || !S_ISREG(sb.st_mode)) {
    std::cerr << "input file '" << input_filename
              << "' does not exist or is not a file." << std::endl;

    return 1;
  }

  // The total size of the input.
  const off_t input_size = sb.st_size;

  const std::vector<std::string> vocab =
      generate_vocab ? CreateVocabulary(input_filename, shard_size,
                                        min_vocab_count, max_vocab_size)
                     : ReadVocabulary(vocab_filename);

  if (!vocab.size()) {
    std::cerr << "Empty vocabulary." << std::endl;
    return 1;
  }

  std::cout << "Generating Swivel co-occurrence data into " << output_dirname
            << std::endl;

  std::cout << "Shard size: " << shard_size << "x" << shard_size << std::endl;
  std::cout << "Vocab size: " << vocab.size() << std::endl;

  // Write the vocabulary files into  the output directory.
  WriteVocabulary(vocab, output_dirname);

  const int num_shards = vocab.size() / shard_size;
  CoocBuffer coocbuf(output_dirname, num_shards, shard_size);

  // Build a mapping from the token to its position in the vocabulary file.
  std::unordered_map<std::string, int> token_to_id_map;
  for (int i = 0; i < static_cast<int>(vocab.size()); ++i)
    token_to_id_map[vocab[i]] = i;

  // Compute the co-occurrences
  std::vector<pthread_t> threads;
  std::vector<CoocCounter*> counters;
  const off_t nbytes_per_thread = input_size / num_threads;

  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) {
    std::cerr << "unable to initalize pthreads" << std::endl;
    return 1;
  }

  for (int i = 0; i < num_threads; ++i) {
    // We could make this smarter and look around for newlines.  But
    // realistically that's not going to change things much.
    const off_t start = i * nbytes_per_thread;
    const off_t end =
        i < num_threads - 1 ? (i + 1) * nbytes_per_thread : input_size;

    CoocCounter *counter = new CoocCounter(
        input_filename, start, end, window_size, token_to_id_map, &coocbuf);

    counters.push_back(counter);

    pthread_t thread;
    pthread_create(&thread, &attr, CoocCounter::Run, counter);

    threads.push_back(thread);
  }

  // Wait for threads to finish and collect marginals.
  std::vector<double> marginals(vocab.size());
  for (int i = 0; i < num_threads; ++i) {
    pthread_join(threads[i], 0);

    const std::vector<double>& counter_marginals = counters[i]->Marginals();
    for (int j = 0; j < static_cast<int>(vocab.size()); ++j)
      marginals[j] += counter_marginals[j];

    delete counters[i];
  }

  std::cout << "writing marginals..." << std::endl;
  WriteMarginals(marginals, output_dirname);

  std::cout << "writing shards..." << std::endl;
  coocbuf.WriteShards();

  return 0;
}
