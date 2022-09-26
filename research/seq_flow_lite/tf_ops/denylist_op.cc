/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tf_ops/skipgram_finder.h"  // seq_flow_lite
#include "tf_ops/subsequence_finder.h"  // seq_flow_lite

namespace seq_flow_lite {

using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::errors::InvalidArgument;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

// Description of the outputs and attributes for the Denylist ops.
const char kDescription[] = R"(
output: A floating point tensor that contains a prediction vector for each
  input string.  The vector will either be:
  * [1, 1, ..., 0, 0, ...] if no denylisted skipgrams are found.
    (All negative categories are 1.0 and all positive categories are 0.0.)
  * an indicator vector if any denylisted skipgrams are found.
    (0.0 if no skipgrams belonging to the category were found and 1.0 otherwise)

max_skip_size: The maximum number of tokens that can be skipped when generating
  skipgrams.

denylist: A string vector containing denylisted skipgrams.

denylist_category: An int32 vector containing the category of the corresponding
  skipgram in the denylist.

categories: An int32 scalar.  This is the total number of categories.
  All categories in denylist_category must be in [0, categories).

negative_categories: An int32 scalar.  The total number of categories that
  should be set if no entries in the denylist are triggered.  These
  negative categories are assumed to be [0, negative_categories).
)";

// The base class for all Denylist ops.  It does two things:
// 1) It defines the output tensor of the op and it defines the attributes
//    needed to specify the denylist and convert denylist categories into
//    output vectors.
// 2) It defines a Compute() function.  The compute function is responsible
//    for filling in the output tensor, while the subclass is responsible
//    for processing the input.
class DenylistOpBase : public OpKernel {
 public:
  explicit DenylistOpBase(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("categories", &categories_));
    OP_REQUIRES_OK(context, context->GetAttr("negative_categories",
                                             &negative_categories_));

    OP_REQUIRES(context, categories_ > 0,
                InvalidArgument("Number of categories (", categories_,
                                ") must be positive."));
    OP_REQUIRES(
        context, negative_categories_ >= 0,
        InvalidArgument("Number of negative_categories (", negative_categories_,
                        ") must be non-negative."));
    OP_REQUIRES(context, negative_categories_ < categories_,
                InvalidArgument("Number of categories (", categories_,
                                ") must be greater than the "
                                "number of negative_categories (",
                                negative_categories_, ")."));

    OP_REQUIRES_OK(context, context->GetAttr("max_skip_size", &max_skip_size_));

    OP_REQUIRES_OK(context, context->GetAttr("denylist", &denylist_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("denylist_category", &denylist_category_));
    OP_REQUIRES(context, denylist_.size() == denylist_category_.size(),
                InvalidArgument("denylist length (", denylist_.size(),
                                ") != denylist_category length (",
                                denylist_category_.size(), ")"));
    int max =
        *std::max_element(denylist_category_.begin(), denylist_category_.end());
    OP_REQUIRES(context, max < categories_,
                InvalidArgument("max element of denylist_category (", max,
                                ") >= categories (", categories_, ")"));
    int min =
        *std::min_element(denylist_category_.begin(), denylist_category_.end());
    OP_REQUIRES(
        context, min >= 0,
        InvalidArgument("min element of denylist_category (", min, ") < 0"));
  }

  void Compute(OpKernelContext* context) override {
    auto compute_context = InitializeComputeContext(context);
    if (compute_context == nullptr) {
      return;
    }
    auto context_cleaner = absl::MakeCleanup([this, compute_context] {
      this->FinalizeComputeContext(compute_context);
    });

    Tensor* output_tensor;
    TensorShape output_shape = InputStringsShape(compute_context);
    output_shape.AddDim(categories_);
    OP_REQUIRES_OK(context, context->allocate_output("output", output_shape,
                                                     &output_tensor));
    auto output_values = output_tensor->flat<float>();

    for (int i = 0; i < NumInputStrings(compute_context); i++) {
      auto category = GetCategories(i, compute_context);
      int base_index = i * categories_;
      if (category.empty()) {
        for (int j = 0; j < categories_; j++) {
          output_values(base_index + j) = j < negative_categories_ ? 1.0 : 0.0;
        }
      } else {
        for (int j = 0; j < categories_; j++) {
          output_values(base_index + j) = category.contains(j) ? 1.0 : 0.0;
        }
      }
    }
  }

 protected:
  int max_skip_size() { return max_skip_size_; }
  int denylist_size() { return denylist_.size(); }
  const std::string& denylist(int i) { return denylist_[i]; }
  int32_t denylist_category(int i) { return denylist_category_[i]; }

 private:
  // Called at the beginning of Compute().  This function should process
  // the input and return a context object that can be used to identify
  // the denylist categories of each input string.
  virtual void* InitializeComputeContext(OpKernelContext* context) = 0;

  // Called at the end of Compute().  Frees the context object.
  virtual void FinalizeComputeContext(void* context) = 0;

  // Returns the shape of the input tensor, if it only consisted of strings.
  // If the input tensor is strings, this is the shape of the input tensor.
  // If the input tensor is tokens, this is the shape of the input tensor,
  // minus the innermost dimension.
  virtual TensorShape InputStringsShape(void* context) = 0;

  // Returns the number of strings in the input tensor.
  virtual int NumInputStrings(void* context) = 0;

  // Returns the denylist categories of the index-th string.
  virtual absl::flat_hash_set<int> GetCategories(int index, void* context) = 0;

  int32_t categories_;
  int32_t negative_categories_;

  int max_skip_size_;
  std::vector<std::string> denylist_;
  std::vector<int32_t> denylist_category_;
};

// A base class for Denylist ops that expect a string tensor input.
class StringDenylistOp : public DenylistOpBase {
 public:
  explicit StringDenylistOp(OpKernelConstruction* context)
      : DenylistOpBase(context) {}

 private:
  void* InitializeComputeContext(OpKernelContext* context) override {
    const Tensor* input_tensor;
    auto status = context->input("input", &input_tensor);
    if (!status.ok()) {
      context->CtxFailureWithWarning(__FILE__, __LINE__, status);
      return nullptr;
    }
    return new ComputeContext(input_tensor);
  }
  void FinalizeComputeContext(void* context) override {
    delete static_cast<ComputeContext*>(context);
  }
  TensorShape InputStringsShape(void* context) override {
    return static_cast<ComputeContext*>(context)->input_tensor->shape();
  }
  int NumInputStrings(void* context) override {
    return static_cast<ComputeContext*>(context)->input_tensor_values.size();
  }
  absl::flat_hash_set<int> GetCategories(int index, void* context) override {
    return FindTerms(
        static_cast<ComputeContext*>(context)->input_tensor_values(index));
  }

  struct ComputeContext {
    ComputeContext(const Tensor* input_tensor)
        : input_tensor(input_tensor),
          input_tensor_values(input_tensor->flat<::tensorflow::tstring>()) {}

    const Tensor* input_tensor;
    ::tensorflow::TTypes<::tensorflow::tstring>::ConstFlat input_tensor_values;
  };

  // Returns the set of denylist categories for the input string.
  virtual absl::flat_hash_set<int> FindTerms(const std::string& input) = 0;
};

// A denylist op that uses the SkipgramFinder on string inputs.
class SkipgramDenylistOp : public StringDenylistOp {
 public:
  explicit SkipgramDenylistOp(OpKernelConstruction* context)
      : StringDenylistOp(context) {
    skipgram_finder_ = std::make_unique<SkipgramFinder>(max_skip_size());
    for (int i = 0; i < denylist_size(); i++) {
      skipgram_finder_->AddSkipgram(denylist(i), denylist_category(i));
    }
  }

 private:
  absl::flat_hash_set<int> FindTerms(const std::string& input) override {
    return skipgram_finder_->FindSkipgrams(input);
  }

  std::unique_ptr<SkipgramFinder> skipgram_finder_;
};

REGISTER_KERNEL_BUILDER(
    Name("SkipgramDenylist").Device(::tensorflow::DEVICE_CPU),
    SkipgramDenylistOp);

// Shape inference function for Denylist ops with string inputs.
Status StringDenylistShapeFn(InferenceContext* context) {
  int32_t categories;
  TF_RETURN_IF_ERROR(context->GetAttr("categories", &categories));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(context->Concatenate(
      context->input(0), context->MakeShape({categories}), &output_shape));
  context->set_output(0, output_shape);
  return ::tensorflow::OkStatus();
}

REGISTER_OP("SkipgramDenylist")
    .Input("input: string")
    .Output("output: float")
    .Attr("max_skip_size: int")
    .Attr("denylist: list(string)")
    .Attr("denylist_category: list(int)")
    .Attr("categories: int")
    .Attr("negative_categories: int")
    .SetShapeFn(StringDenylistShapeFn)
    .Doc(absl::StrCat("Generates dense prediction vectors for input strings "
                      "using a skipgram denylist.",
                      "\n\n", "input: A string tensor.", "\n\n", kDescription));

// A Denylist op that uses the SubsequenceFinder on string inputs.
class SubsequenceDenylistOp : public StringDenylistOp {
 public:
  explicit SubsequenceDenylistOp(OpKernelConstruction* context)
      : StringDenylistOp(context) {
    subsequence_finder_ = std::make_unique<SubsequenceFinder>(max_skip_size());
    for (int i = 0; i < denylist_size(); i++) {
      subsequence_finder_->AddSubsequence(denylist(i), denylist_category(i));
    }
  }

 private:
  absl::flat_hash_set<int> FindTerms(const std::string& input) override {
    return subsequence_finder_->FindSubsequences(input);
  }

  std::unique_ptr<SubsequenceFinder> subsequence_finder_;
};

REGISTER_KERNEL_BUILDER(
    Name("SubsequenceDenylist").Device(::tensorflow::DEVICE_CPU),
    SubsequenceDenylistOp);

REGISTER_OP("SubsequenceDenylist")
    .Input("input: string")
    .Output("output: float")
    .Attr("max_skip_size: int")
    .Attr("denylist: list(string)")
    .Attr("denylist_category: list(int)")
    .Attr("categories: int")
    .Attr("negative_categories: int")
    .SetShapeFn(StringDenylistShapeFn)
    .Doc(absl::StrCat("Generates dense prediction vectors for inputs using a "
                      "subsequence denylist.",
                      "\n\n", "input: A string tensor.", "\n\n", kDescription));

// A denylist op that uses the SkipgramFinder on tokenized string inputs.
// The inputs are a pair of tensors: a token tensor of type string and
// a token count tensor of type T.
template <typename T>
class TokenizedDenylistOp : public DenylistOpBase {
 public:
  explicit TokenizedDenylistOp(OpKernelConstruction* context)
      : DenylistOpBase(context) {
    skipgram_finder_ = std::make_unique<SkipgramFinder>(max_skip_size());
    for (int i = 0; i < denylist_size(); i++) {
      skipgram_finder_->AddSkipgram(denylist(i), denylist_category(i));
    }
  }

 private:
  void* InitializeComputeContext(OpKernelContext* context) override {
    const Tensor* input_tensor;
    {
      auto status = context->input("input", &input_tensor);
      if (!status.ok()) {
        context->CtxFailureWithWarning(__FILE__, __LINE__, status);
        return nullptr;
      }
    }

    const Tensor* token_count_tensor;
    {
      auto status = context->input("token_count", &token_count_tensor);
      if (!status.ok()) {
        context->CtxFailureWithWarning(__FILE__, __LINE__, status);
        return nullptr;
      }
    }

    return new ComputeContext(input_tensor, token_count_tensor);
  }
  void FinalizeComputeContext(void* context) override {
    delete static_cast<ComputeContext*>(context);
  }
  TensorShape InputStringsShape(void* context) override {
    return static_cast<ComputeContext*>(context)->shape;
  }
  int NumInputStrings(void* context) override {
    return static_cast<ComputeContext*>(context)->size;
  }
  absl::flat_hash_set<int> GetCategories(int index, void* x) override {
    ComputeContext* context = static_cast<ComputeContext*>(x);

    int64_t num_tokens = context->token_count_flat(index);
    std::vector<absl::string_view> tokens;
    tokens.reserve(num_tokens);

    int64_t start = index * context->max_tokens;
    for (int64_t i = start; i < start + num_tokens; i++) {
      tokens.emplace_back(context->token_flat(i).data(),
                          context->token_flat(i).size());
    }
    return skipgram_finder_->FindSkipgrams(tokens);
  }

  struct ComputeContext {
    ComputeContext(const Tensor* token_tensor, const Tensor* token_count_tensor)
        : token_flat(token_tensor->flat<::tensorflow::tstring>()),
          token_count_flat(token_count_tensor->flat<T>()) {
      shape = token_tensor->shape();
      max_tokens = shape.dim_size(shape.dims() - 1);
      shape.RemoveLastDims(1);
      size = 1;
      for (int64_t i = 0; i < shape.dims(); i++) {
        size = size * shape.dim_size(i);
      }
    }

    const typename ::tensorflow::TTypes<::tensorflow::tstring>::ConstFlat
        token_flat;
    const typename ::tensorflow::TTypes<T>::ConstFlat token_count_flat;
    TensorShape shape;
    int64_t size;
    int64_t max_tokens;
  };

  std::unique_ptr<SkipgramFinder> skipgram_finder_;
};

REGISTER_KERNEL_BUILDER(Name("TokenizedDenylist")
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32_t>("Ttoken_count"),
                        TokenizedDenylistOp<int32_t>);
REGISTER_KERNEL_BUILDER(Name("TokenizedDenylist")
                            .Device(::tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64_t>("Ttoken_count"),
                        TokenizedDenylistOp<int64_t>);

// Shape inference function for Denylist ops with tokenized string inputs.
Status TokenizedDenylistShapeFn(InferenceContext* context) {
  int32_t categories;
  TF_RETURN_IF_ERROR(context->GetAttr("categories", &categories));

  ShapeHandle string_tensor_shape;
  TF_RETURN_IF_ERROR(
      context->Subshape(context->input(0), 0, -1, &string_tensor_shape));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(context->Concatenate(
      string_tensor_shape, context->MakeShape({categories}), &output_shape));
  context->set_output(0, output_shape);

  return ::tensorflow::OkStatus();
}

REGISTER_OP("TokenizedDenylist")
    .Input("input: string")
    .Input("token_count: Ttoken_count")
    .Output("output: float")
    .Attr("max_skip_size: int")
    .Attr("denylist: list(string)")
    .Attr("denylist_category: list(int)")
    .Attr("categories: int")
    .Attr("negative_categories: int")
    .Attr("Ttoken_count: {int32, int64}")
    .SetShapeFn(TokenizedDenylistShapeFn)
    .Doc(absl::StrCat("Generates dense prediction vectors for tokens using a "
                      "skipgram denylist.",
                      "\n\n", "input: A string tensor of tokens.", "\n\n",
                      kDescription));

}  // namespace seq_flow_lite
