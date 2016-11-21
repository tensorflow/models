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

#define EIGEN_USE_THREADS

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "syntaxnet/sparse.pb.h"
#include "syntaxnet/utils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"

using tensorflow::DEVICE_CPU;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_STRING;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

namespace syntaxnet {

// Operator to unpack ids and weights stored in SparseFeatures proto.
class UnpackSparseFeatures : public OpKernel {
 public:
  explicit UnpackSparseFeatures(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->MatchSignature(
                                {DT_STRING}, {DT_INT32, DT_INT64, DT_FLOAT}));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    OP_REQUIRES(context, IsLegacyVector(input.shape()),
                InvalidArgument("input should be a vector."));

    const int64 n = input.NumElements();
    const auto input_vec = input.flat<string>();
    SparseFeatures sf;
    int output_size = 0;
    std::vector<std::pair<int64, float> > id_and_weight;

    // Guess that we'll be averaging a handful of ids per SparseFeatures record.
    id_and_weight.reserve(n * 4);
    std::vector<int> num_ids(n);
    for (int64 i = 0; i < n; ++i) {
      OP_REQUIRES(context, sf.ParseFromString(input_vec(i)),
                  InvalidArgument("Couldn't parse as SparseFeature"));
      OP_REQUIRES(context,
                  sf.weight_size() == 0 || sf.weight_size() == sf.id_size(),
                  InvalidArgument(tensorflow::strings::StrCat(
                      "Incorrect number of weights", sf.DebugString())));
      int n_ids = sf.id_size();
      num_ids[i] = n_ids;
      output_size += n_ids;
      for (int j = 0; j < n_ids; j++) {
        float w = (sf.weight_size() > 0) ? sf.weight(j) : 1.0f;
        id_and_weight.push_back(std::make_pair(sf.id(j), w));
      }
    }

    Tensor *indices_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_size}), &indices_t));
    Tensor *ids_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, TensorShape({output_size}), &ids_t));
    Tensor *weights_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, TensorShape({output_size}), &weights_t));

    auto indices = indices_t->vec<int32>();
    auto ids = ids_t->vec<int64>();
    auto weights = weights_t->vec<float>();
    int c = 0;
    for (int64 i = 0; i < n; ++i) {
      for (int j = 0; j < num_ids[i]; ++j) {
        indices(c) = i;
        ids(c) = id_and_weight[c].first;
        weights(c) = id_and_weight[c].second;
        ++c;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("UnpackSparseFeatures").Device(DEVICE_CPU),
                        UnpackSparseFeatures);

}  // namespace syntaxnet
