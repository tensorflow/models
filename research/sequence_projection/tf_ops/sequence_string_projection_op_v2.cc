/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tf_ops/projection_normalizer_util.h"  // sequence_projection
#include "tf_ops/projection_util.h"  // sequence_projection
#include "tf_ops/text_distorter.h"  // sequence_projection

using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::uint64;
using ::tensorflow::errors::InvalidArgument;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

constexpr char kBeginTokenTSP[] = "<BOS>";
constexpr char kEndTokenTSP[] = "<EOS>";
constexpr float kMappingTable[4] = {0, 1, -1, 0};
constexpr int kIncrement = 32;

class SequenceStringProjectionOpV2 : public OpKernel {
 public:
  explicit SequenceStringProjectionOpV2(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("feature_size", &feature_size_));
    hasher_ = absl::make_unique<Hasher>(feature_size_);

    float distortion_probability = 0.0;
    OP_REQUIRES_OK(context, context->GetAttr("distortion_probability",
                                             &distortion_probability));
    text_distorter_ = absl::make_unique<TextDistorter>(distortion_probability);

    OP_REQUIRES_OK(context, context->GetAttr("vocabulary", &vocabulary_));
    unicode_handler_ = absl::make_unique<ProjectionUnicodeHandler>(vocabulary_);

    bool add_bos_tag;
    OP_REQUIRES_OK(context, context->GetAttr("add_bos_tag", &add_bos_tag));
    bos_tag_ = add_bos_tag ? 1 : 0;

    bool add_eos_tag;
    OP_REQUIRES_OK(context, context->GetAttr("add_eos_tag", &add_eos_tag));
    eos_tag_ = add_eos_tag ? 1 : 0;

    bool normalize_repetition;
    OP_REQUIRES_OK(context, context->GetAttr("normalize_repetition",
                                             &normalize_repetition));
    if (normalize_repetition) {
      projection_normalizer_ = absl::make_unique<ProjectionNormalizer>(
          std::string(), normalize_repetition);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input_tensor->shape()),
                InvalidArgument("`input` must be a matrix, got shape: ",
                                input_tensor->shape().DebugString()));
    auto input_matrix = input_tensor->matrix<::tensorflow::tstring>();
    const int64 batch_size = input_matrix.dimension(0);
    const int64 max_seq_len = input_matrix.dimension(1);

    const Tensor* seq_len;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_length", &seq_len));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(seq_len->shape()),
        InvalidArgument("`sequence_length` must be a vector, got shape: ",
                        seq_len->shape().DebugString()));
    auto seq_len_vector = seq_len->vec<int32>();

    OP_REQUIRES(
        ctx, seq_len_vector.size() == batch_size,
        InvalidArgument("`sequence_length` should have batch size number "
                        "of elements, got size ",
                        seq_len_vector.size(), ", batch size is ", batch_size));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
                 "projection",
                 {batch_size, bos_tag_ + max_seq_len + eos_tag_, feature_size_},
                 &output_tensor));
    float* projection = &output_tensor->flat<float>()(0);

    std::vector<uint64_t> hash_codes;
    for (int64 i = 0; i < batch_size; ++i) {
      const int64 num_tokens = seq_len_vector(i);
      OP_REQUIRES(ctx, num_tokens > 0,
                  InvalidArgument(
                      "`sequence_length` should have values greater than 0"));
      OP_REQUIRES(ctx, num_tokens <= max_seq_len,
                  InvalidArgument("`sequence_length` should have values less "
                                  "than or equal to max_seq_len"));

      int64 offset0 = i * (bos_tag_ + max_seq_len + eos_tag_) * feature_size_;
      for (int64 j = -bos_tag_; j < num_tokens + eos_tag_; ++j) {
        std::string word;
        if (j < 0) {
          word = kBeginTokenTSP;
        } else if (j < num_tokens) {
          auto token = std::pair<const char*, int32>(input_matrix(i, j).data(),
                                                     input_matrix(i, j).size());
          auto uword = icu::UnicodeString::fromUTF8(
              unicode_handler_->LowerCaseUTF8WithSupportedUnicodes(token));
          word = text_distorter_->DistortText(&uword);
          if (projection_normalizer_) {
            word = projection_normalizer_->Normalize(word.data(), word.size(),
                                                     SIZE_MAX);
          }
        } else {
          word = kEndTokenTSP;
        }
        hasher_->GetHashCodes(word, &hash_codes);
        for (int hindex = 0, k = 0; hindex < hash_codes.size(); hindex++) {
          auto hash = hash_codes[hindex];
          for (int kmax = std::min(k + kIncrement, feature_size_); k < kmax;) {
            projection[offset0 + k++] = kMappingTable[hash & 0x3];
            hash >>= 2;
          }
        }
        offset0 += feature_size_;
      }
      const int fill_length = (max_seq_len - num_tokens) * feature_size_;
      float* fill_start = projection + offset0;
      std::fill(fill_start, fill_start + fill_length, 0.0f);
    }
  }

 private:
  int32 feature_size_;
  std::unique_ptr<Hasher> hasher_;
  std::unique_ptr<TextDistorter> text_distorter_;
  std::unique_ptr<ProjectionUnicodeHandler> unicode_handler_;
  std::unique_ptr<ProjectionNormalizer> projection_normalizer_;
  std::string vocabulary_;
  int eos_tag_;
  int bos_tag_;
};

REGISTER_KERNEL_BUILDER(
    Name("SequenceStringProjectionV2").Device(::tensorflow::DEVICE_CPU),
    SequenceStringProjectionOpV2);

REGISTER_OP("SequenceStringProjectionV2")
    .Input("input: string")
    .Input("sequence_length: int32")
    .Output("projection: float32")
    .Attr("feature_size: int")
    .Attr("distortion_probability: float = 0.0")
    .Attr("vocabulary: string = ''")
    .Attr("add_bos_tag: bool = False")
    .Attr("add_eos_tag: bool = False")
    .Attr("normalize_repetition: bool = False")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      DimensionHandle size;

      int32 feature_size;
      TF_RETURN_IF_ERROR(c->GetAttr("feature_size", &feature_size));
      const int kMaxFeatureSize = 4096;
      CHECK_GT(feature_size, 0);
      CHECK_LE(feature_size, kMaxFeatureSize);
      ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(c->Concatenate(
          c->input(0), c->MakeShape({feature_size}), &output_shape));
      c->set_output(0, output_shape);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
This op referred to as Ternary Sequence String Projection Op V2 (TSPV2),
works with presegmented string `input`. It fingerprints each token using murmur
hash and extracts bit features from the fingerprint that maps every 2 bits to
the ternary output {-1, 0, 1}. This effectively turns a batch of text segments
into a ternary rank 3 tensor (in float format) of shape
[batch size, max sequence length, requested number of features].

Input(s):
- input: A string tensor with [batch size, max sequence length] tokens.
- sequence_length: A vector with batch size number of integers, where each
    integer is in (0, max sequence length], and represents the number of valid
    text segments in each batch entry.

Attribute(s):
- feature_size: Length of the ternary vector generated for each token.
- distortion_probability: When non zero distort the input tokens with this
    probability. Helps as a regularization method when training data set is
    small.
- vocabulary: When not empty provides a list of unique unicode characters that
    will be allowed in the input text before fingerprinting. Expressed another
    way the vocabulary is an optional character allowlist for the
    input tokens. It helps normalize the text.
- add_bos_tag: When true inserts a begin of sentence tag.
- add_eos_tag: When true inserts a end of sentence tag.
- normalize_repetition: When true normalizes repetition in text tokens before
    fingerprinting.

Output(s):
- projection: Floating point tensor with ternary values of shape
    [batch size, max sequence length, requested number of features].
)doc");
