// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "dragnn/runtime/myelin/myelin_library.h"

#include <vector>

#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Tests that PreMultipliedEmbeddings does nothing on an empty Flow.
TEST(PreMultipliedEmbeddingsTest, DoesNothingOnEmptyFlow) {
  sling::myelin::Flow flow;
  PreMultipliedEmbeddings transformer;
  EXPECT_FALSE(transformer.Transform(&flow));
}

// Tests that PreMultipliedEmbeddings can rearrange a MatMul of a Gather into a
// Gather of a pre-multiplied matrix.
TEST(PreMultipliedEmbeddingsTest, AppliesPreMultiplication) {
  sling::myelin::Flow flow;
  sling::myelin::Flow::Function *function = flow.AddFunction("test_function");
  sling::myelin::Flow::Variable *indices =
      flow.AddVariable("indices", sling::myelin::DT_INT32, {1});
  sling::myelin::Flow::Variable *embeddings =
      flow.AddVariable("embeddings", sling::myelin::DT_FLOAT, {10, 20});
  sling::myelin::Flow::Variable *gathered =
      flow.AddVariable("gathered", sling::myelin::DT_FLOAT, {1, 20});
  sling::myelin::Flow::Variable *weights =
      flow.AddVariable("weights", sling::myelin::DT_FLOAT, {20, 30});
  sling::myelin::Flow::Variable *output =
      flow.AddVariable("output", sling::myelin::DT_FLOAT, {1, 30});
  flow.AddOperation(function, "gather", "Gather", {embeddings, indices},
                    {gathered});
  flow.AddOperation(function, "matmul", "MatMul", {gathered, weights},
                    {output});

  // Attach constant data to the matrices.
  const std::vector<float> floats(20 * 30);  // big enough for both
  embeddings->SetData(floats.data(), 10 * 20 * sizeof(float));
  weights->SetData(floats.data(), 20 * 30 * sizeof(float));

  PreMultipliedEmbeddings transformer;
  ASSERT_TRUE(transformer.Transform(&flow));

  sling::myelin::Flow::Variable *product = flow.Var("embeddings/weights");
  ASSERT_NE(product, nullptr);
  ASSERT_EQ(product->rank(), 2);
  EXPECT_EQ(product->dim(0), 10);
  EXPECT_EQ(product->dim(1), 30);

  sling::myelin::Flow::Operation *pre_multiply =
      flow.Op("embeddings/weights/PreMultiply");
  ASSERT_NE(pre_multiply, nullptr);
  ASSERT_EQ(pre_multiply->indegree(), 2);
  ASSERT_EQ(pre_multiply->outdegree(), 1);

  EXPECT_EQ(pre_multiply->type, "MatMul");
  EXPECT_EQ(pre_multiply->inputs[0], embeddings);
  EXPECT_EQ(pre_multiply->inputs[1], weights);
  EXPECT_EQ(pre_multiply->outputs[0], product);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
