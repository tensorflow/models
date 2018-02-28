/* -*- Mode: C++ -*- */

/*
 * Copyright 2016 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Computes embedding performance on analogy tasks.  Accepts as input one or
 * more files containing four words per line (A B C D), and determines if:
 *
 *   vec(C) - vec(A) + vec(B) ~= vec(D)
 *
 * Cosine distance in the embedding space is used to retrieve neighbors. Any
 * missing vocabulary items are scored as losses.
 */
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static const char usage[] = R"(
Performs analogy testing of embedding vectors.

Usage:

  analogy --embeddings <embeddings> --vocab <vocab> eval1.tab ...

Options:

  --embeddings <filename>
    The file containing the binary embedding vectors to evaluate.

  --vocab <filename>
    The vocabulary file corresponding to the embedding vectors.

  --nthreads <integer>
    The number of evaluation threads to run (default: 8)
)";

// Reads the vocabulary file into a map from token to vector index.
static std::unordered_map<std::string, int> ReadVocab(
    const std::string& vocab_filename) {
  std::unordered_map<std::string, int> vocab;
  std::ifstream fin(vocab_filename);

  int index = 0;
  for (std::string token; std::getline(fin, token); ++index) {
    auto n = token.find('\t');
    if (n != std::string::npos) token = token.substr(n);

    vocab[token] = index;
  }

  return vocab;
}

// An analogy query: "A is to B as C is to D".
typedef std::tuple<int, int, int, int> AnalogyQuery;

std::vector<AnalogyQuery> ReadQueries(
    const std::string &filename,
    const std::unordered_map<std::string, int> &vocab, int *total) {
  std::ifstream fin(filename);

  std::vector<AnalogyQuery> queries;
  int lineno = 0;
  while (1) {
    // Read the four words.
    std::string words[4];
    int nread = 0;
    for (int i = 0; i < 4; ++i) {
      fin >> words[i];
      if (!words[i].empty()) ++nread;
    }

    ++lineno;
    if (nread == 0) break;

    if (nread < 4) {
      std::cerr << "expected four words at line " << lineno << std::endl;
      break;
    }

    // Look up each word's index.
    int ixs[4], nvalid;
    for (nvalid = 0; nvalid < 4; ++nvalid) {
      std::unordered_map<std::string, int>::const_iterator it =
          vocab.find(words[nvalid]);

      if (it == vocab.end()) break;

      ixs[nvalid] = it->second;
    }

    // If we don't have all the words, count it as a loss.
    if (nvalid >= 4)
      queries.push_back(std::make_tuple(ixs[0], ixs[1], ixs[2], ixs[3]));
  }

  *total = lineno;
  return queries;
}


// A thread that evaluates some fraction of the analogies.
class AnalogyEvaluator {
 public:
  // Creates a new Analogy evaluator for a range of analogy queries.
  AnalogyEvaluator(std::vector<AnalogyQuery>::const_iterator begin,
                   std::vector<AnalogyQuery>::const_iterator end,
                   const float *embeddings, const int num_embeddings,
                   const int dim)
      : begin_(begin),
        end_(end),
        embeddings_(embeddings),
        num_embeddings_(num_embeddings),
        dim_(dim) {}

  // A thunk for pthreads.
  static void* Run(void *param) {
    AnalogyEvaluator *self = static_cast<AnalogyEvaluator*>(param);
    self->Evaluate();
    return nullptr;
  }

  // Evaluates the analogies.
  void Evaluate();

  // Returns the number of correct analogies after evaluation is complete.
  int GetNumCorrect() const { return correct_; }

 protected:
  // The beginning of the range of queries to consider.
  std::vector<AnalogyQuery>::const_iterator begin_;

  // The end of the range of queries to consider.
  std::vector<AnalogyQuery>::const_iterator end_;

  // The raw embedding vectors.
  const float *embeddings_;

  // The number of embedding vectors.
  const int num_embeddings_;

  // The embedding vector dimensionality.
  const int dim_;

  // The number of correct analogies.
  int correct_;
};


void AnalogyEvaluator::Evaluate() {
  float* sum = new float[dim_];

  correct_ = 0;
  for (auto query = begin_; query < end_; ++query) {
    const float* vec;
    int a, b, c, d;
    std::tie(a, b, c, d) = *query;

    // Compute C - A + B.
    vec = embeddings_ + dim_ * c;
    for (int i = 0; i < dim_; ++i) sum[i] = vec[i];

    vec = embeddings_ + dim_ * a;
    for (int i = 0; i < dim_; ++i) sum[i] -= vec[i];

    vec = embeddings_ + dim_ * b;
    for (int i = 0; i < dim_; ++i) sum[i] += vec[i];

    // Find the nearest neighbor that isn't one of the query words.
    int best_ix = -1;
    float best_dot = -1.0;
    for (int i = 0; i < num_embeddings_; ++i) {
      if (i == a || i == b || i == c) continue;

      vec = embeddings_ + dim_ * i;

      float dot = 0;
      for (int j = 0; j < dim_; ++j) dot += vec[j] * sum[j];

      if (dot > best_dot) {
        best_ix = i;
        best_dot = dot;
      }
    }

    // The fourth word is the answer; did we get it right?
    if (best_ix == d) ++correct_;
  }

  delete[] sum;
}


int main(int argc, char *argv[]) {
  if (argc <= 1) {
    printf(usage);
    return 2;
  }

  std::string embeddings_filename, vocab_filename;
  int nthreads = 8;

  std::vector<std::string> input_filenames;
  std::vector<std::tuple<int, int, int, int>> queries;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--embeddings") {
      if (++i >= argc) goto argmissing;
      embeddings_filename = argv[i];
    } else if (arg == "--vocab") {
      if (++i >= argc) goto argmissing;
      vocab_filename = argv[i];
    } else if (arg == "--nthreads") {
      if (++i >= argc) goto argmissing;
      if ((nthreads = atoi(argv[i])) <= 0) goto badarg;
    } else if (arg == "--help") {
      std::cout << usage << std::endl;
      return 0;
    } else if (arg[0] == '-') {
      std::cerr << "unknown option: '" << arg << "'" << std::endl;
      return 2;
    } else {
      input_filenames.push_back(arg);
    }

    continue;

  argmissing:
    std::cerr << "missing value for '" << argv[i - 1] << "' (--help for help)"
              << std::endl;
    return 2;

  badarg:
    std::cerr << "invalid value '" << argv[i] << "' for '" << argv[i - 1]
              << "' (--help for help)" << std::endl;

    return 2;
  }

  // Read the vocabulary.
  std::unordered_map<std::string, int> vocab = ReadVocab(vocab_filename);
  if (!vocab.size()) {
    std::cerr << "unable to read vocabulary file '" << vocab_filename << "'"
              << std::endl;
    return 1;
  }

  const int n = vocab.size();

  // Read the vectors.
  int fd;
  if ((fd = open(embeddings_filename.c_str(), O_RDONLY)) < 0) {
    std::cerr << "unable to open embeddings file '" << embeddings_filename
              << "'" << std::endl;
    return 1;
  }

  off_t nbytes = lseek(fd, 0, SEEK_END);
  if (nbytes == -1) {
    std::cerr << "unable to determine file size for '" << embeddings_filename
              << "'" << std::endl;
    return 1;
  }

  if (nbytes % (sizeof(float) * n) != 0) {
    std::cerr << "'" << embeddings_filename
              << "' has a strange file size; expected it to be "
                 "a multiple of the vocabulary size"
              << std::endl;

    return 1;
  }

  const int dim = nbytes / (sizeof(float) * n);
  float *embeddings = static_cast<float *>(malloc(nbytes));
  lseek(fd, 0, SEEK_SET);
  if (read(fd, embeddings, nbytes) < nbytes) {
    std::cerr << "unable to read embeddings from " << embeddings_filename
              << std::endl;
    return 1;
  }

  close(fd);

  /* Normalize the vectors. */
  for (int i = 0; i < n; ++i) {
    float *vec = embeddings + dim * i;
    float norm = 0;
    for (int j = 0; j < dim; ++j) norm += vec[j] * vec[j];

    norm = sqrt(norm);
    for (int j = 0; j < dim; ++j) vec[j] /= norm;
  }

  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0) {
    std::cerr << "unable to initalize pthreads" << std::endl;
    return 1;
  }

  /* Read each input file. */
  for (const auto filename : input_filenames) {
    int total = 0;
    std::vector<AnalogyQuery> queries =
        ReadQueries(filename.c_str(), vocab, &total);

    const int queries_per_thread = queries.size() / nthreads;
    std::vector<AnalogyEvaluator*> evaluators;
    std::vector<pthread_t> threads;

    for (int i = 0; i < nthreads; ++i) {
      auto begin = queries.begin() + i * queries_per_thread;
      auto end = (i + 1 < nthreads)
                     ? queries.begin() + (i + 1) * queries_per_thread
                     : queries.end();

      AnalogyEvaluator *evaluator =
          new AnalogyEvaluator(begin, end, embeddings, n, dim);

      pthread_t thread;
      pthread_create(&thread, &attr, AnalogyEvaluator::Run, evaluator);
      evaluators.push_back(evaluator);
      threads.push_back(thread);
    }

    for (auto &thread : threads) pthread_join(thread, 0);

    int correct = 0;
    for (const AnalogyEvaluator* evaluator : evaluators) {
      correct += evaluator->GetNumCorrect();
      delete evaluator;
    }

    printf("%0.3f %s\n", static_cast<float>(correct) / total, filename.c_str());
  }

  return 0;
}
