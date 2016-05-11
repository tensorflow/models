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

// Various utilities for handling documents.

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "syntaxnet/base.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/utils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"

using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

namespace syntaxnet {

namespace {

void GetTaskContext(OpKernelConstruction *context, TaskContext *task_context) {
  string file_path, data;
  OP_REQUIRES_OK(context, context->GetAttr("task_context", &file_path));
  OP_REQUIRES_OK(
      context, ReadFileToString(tensorflow::Env::Default(), file_path, &data));
  OP_REQUIRES(context,
              TextFormat::ParseFromString(data, task_context->mutable_spec()),
              InvalidArgument("Could not parse task context at ", file_path));
}

// Outputs the given batch of sentences as a tensor and deletes them.
void OutputDocuments(OpKernelContext *context,
                     vector<Sentence *> *document_batch) {
  const int64 size = document_batch->size();
  Tensor *output;
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, TensorShape({size}), &output));
  for (int64 i = 0; i < size; ++i) {
    output->vec<string>()(i) = (*document_batch)[i]->SerializeAsString();
  }
  utils::STLDeleteElements(document_batch);
}

}  // namespace

class DocumentSource : public OpKernel {
 public:
  explicit DocumentSource(OpKernelConstruction *context) : OpKernel(context) {
    GetTaskContext(context, &task_context_);
    string corpus_name;
    OP_REQUIRES_OK(context, context->GetAttr("corpus_name", &corpus_name));
    OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES(context, batch_size_ > 0,
                InvalidArgument("invalid batch_size provided"));
    corpus_.reset(new TextReader(*task_context_.GetInput(corpus_name)));
  }

  void Compute(OpKernelContext *context) override {
    mutex_lock lock(mu_);
    Sentence *document;
    vector<Sentence *> document_batch;
    while ((document = corpus_->Read()) != NULL) {
      document_batch.push_back(document);
      if (static_cast<int>(document_batch.size()) == batch_size_) {
        OutputDocuments(context, &document_batch);
        OutputLast(context, false);
        return;
      }
    }
    OutputDocuments(context, &document_batch);
    OutputLast(context, true);
  }

 private:
  void OutputLast(OpKernelContext *context, bool last) {
    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &output));
    output->scalar<bool>()() = last;
  }

  // Task context used to configure this op.
  TaskContext task_context_;

  // mutex to synchronize access to Compute.
  mutex mu_;

  std::unique_ptr<TextReader> corpus_;
  string documents_path_;
  int batch_size_;
};

REGISTER_KERNEL_BUILDER(Name("DocumentSource").Device(DEVICE_CPU),
                        DocumentSource);

class DocumentSink : public OpKernel {
 public:
  explicit DocumentSink(OpKernelConstruction *context) : OpKernel(context) {
    GetTaskContext(context, &task_context_);
    string corpus_name;
    OP_REQUIRES_OK(context, context->GetAttr("corpus_name", &corpus_name));
    writer_.reset(new TextWriter(*task_context_.GetInput(corpus_name)));
  }

  void Compute(OpKernelContext *context) override {
    mutex_lock lock(mu_);
    auto documents = context->input(0).vec<string>();
    for (int i = 0; i < documents.size(); ++i) {
      Sentence document;
      OP_REQUIRES(context, document.ParseFromString(documents(i)),
                  InvalidArgument("failed to parse sentence"));
      writer_->Write(document);
    }
  }

 private:
  // Task context used to configure this op.
  TaskContext task_context_;

  // mutex to synchronize access to Compute.
  mutex mu_;

  string documents_path_;
  std::unique_ptr<TextWriter> writer_;
};

REGISTER_KERNEL_BUILDER(Name("DocumentSink").Device(DEVICE_CPU),
                        DocumentSink);

// Sentence filter for filtering out documents where the parse trees are not
// well-formed, i.e. they contain cycles.
class WellFormedFilter : public OpKernel {
 public:
  explicit WellFormedFilter(OpKernelConstruction *context) : OpKernel(context) {
    GetTaskContext(context, &task_context_);
    OP_REQUIRES_OK(context, context->GetAttr("keep_malformed_documents",
                                             &keep_malformed_));
  }

  void Compute(OpKernelContext *context) override {
    auto documents = context->input(0).vec<string>();
    vector<Sentence *> output_documents;
    for (int i = 0; i < documents.size(); ++i) {
      Sentence *document = new Sentence;
      OP_REQUIRES(context, document->ParseFromString(documents(i)),
                  InvalidArgument("failed to parse sentence"));
      if (ShouldKeep(*document)) {
        output_documents.push_back(document);
      } else {
        delete document;
      }
    }
    OutputDocuments(context, &output_documents);
  }

 private:
  bool ShouldKeep(const Sentence &doc)  {
    vector<int> visited(doc.token_size(), -1);
    for (int i = 0; i < doc.token_size(); ++i) {
      // Already visited node.
      if (visited[i] != -1) continue;
      int t = i;
      while (t != -1) {
        if (visited[t] == -1) {
          // If it is not visited yet, mark it.
          visited[t] = i;
        } else if (visited[t] < i) {
          // If the index number is smaller than index and not -1, the token has
          // already been visited.
          break;
        } else {
          // Loop detected.
          LOG(ERROR) << "Loop detected in document " << doc.DebugString();
          return keep_malformed_;
        }
        t = doc.token(t).head();
      }
    }
    return true;
  }

 private:
  // Task context used to configure this op.
  TaskContext task_context_;

  bool keep_malformed_;
};

REGISTER_KERNEL_BUILDER(Name("WellFormedFilter").Device(DEVICE_CPU),
                        WellFormedFilter);

// Sentence filter that modifies dependency trees to make them projective. This
// could be made more efficient by looping over sentences instead of the entire
// document. Assumes that the document is well-formed in the sense of having
// no looping dependencies.
//
// Task arguments:
//   bool discard_non_projective (false) : If true, discards documents with
//     non-projective trees instead of projectivizing them.
class ProjectivizeFilter : public OpKernel {
 public:
  explicit ProjectivizeFilter(OpKernelConstruction *context)
      : OpKernel(context) {
    GetTaskContext(context, &task_context_);
    OP_REQUIRES_OK(context, context->GetAttr("discard_non_projective",
                                             &discard_non_projective_));
  }

  void Compute(OpKernelContext *context) override {
    auto documents = context->input(0).vec<string>();
    vector<Sentence *> output_documents;
    for (int i = 0; i < documents.size(); ++i) {
      Sentence *document = new Sentence;
      OP_REQUIRES(context, document->ParseFromString(documents(i)),
                  InvalidArgument("failed to parse sentence"));
      if (Process(document)) {
        output_documents.push_back(document);
      } else {
        delete document;
      }
    }
    OutputDocuments(context, &output_documents);
  }

  bool Process(Sentence *doc) {
    const int num_tokens = doc->token_size();

    // Left and right boundaries for arcs. The left and right ends of an arc are
    // bounded by the arcs that pass over it. If an arc exceeds these bounds it
    // will cross an arc passing over it, making it a non-projective arc.
    vector<int> left(num_tokens);
    vector<int> right(num_tokens);

    // Lift the shortest non-projective arc until the document is projective.
    while (true) {
      // Initialize boundaries to the whole document for all arcs.
      for (int i = 0; i < num_tokens; ++i) {
        left[i] = -1;
        right[i] = num_tokens - 1;
      }

      // Find left and right bounds for each token.
      for (int i = 0; i < num_tokens; ++i) {
        int head_index = doc->token(i).head();

        // Find left and right end of arc.
        int l = std::min(i, head_index);
        int r = std::max(i, head_index);

        // Bound all tokens under the arc.
        for (int j = l + 1; j < r; ++j) {
          if (left[j] < l) left[j] = l;
          if (right[j] > r) right[j] = r;
        }
      }

      // Find deepest non-projective arc.
      int deepest_arc = -1;
      int max_depth = -1;

      // The non-projective arcs are those that exceed their bounds.
      for (int i = 0; i < num_tokens; ++i) {
        int head_index = doc->token(i).head();
        if (head_index == -1) continue;  // any crossing arc must be deeper

        int l = std::min(i, head_index);
        int r = std::max(i, head_index);

        int left_bound = std::max(left[l], left[r]);
        int right_bound = std::min(right[l], right[r]);

        if (l < left_bound || r > right_bound) {
          // Found non-projective arc.
          if (discard_non_projective_) return false;

          // Pick the deepest as the best candidate for lifting.
          int depth = 0;
          int j = i;
          while (j != -1) {
            ++depth;
            j = doc->token(j).head();
          }
          if (depth > max_depth) {
            deepest_arc = i;
            max_depth = depth;
          }
        }
      }

      // If there are no more non-projective arcs we are done.
      if (deepest_arc == -1) return true;

      // Lift non-projective arc.
      int lifted_head = doc->token(doc->token(deepest_arc).head()).head();
      doc->mutable_token(deepest_arc)->set_head(lifted_head);
    }
  }

 private:
  // Task context used to configure this op.
  TaskContext task_context_;

  // Whether or not to throw away non-projective documents.
  bool discard_non_projective_;
};

REGISTER_KERNEL_BUILDER(Name("ProjectivizeFilter").Device(DEVICE_CPU),
                        ProjectivizeFilter);

}  // namespace syntaxnet
