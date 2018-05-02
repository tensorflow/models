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

#include "dragnn/runtime/math/avx_activation_functions.h"

#include <cmath>

#include <chrono>

#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

TEST(AvxActivationFunctionsTest, ExponentialTest) {
  AvxVectorFuzzTest(
      [](AvxFloatVec *vec) { *vec = activations::Exponential(*vec); },
      [](float input_value, float actual) {
        const float inverted = log(actual);
        EXPECT_NEAR(input_value, inverted, 1e-6)
            << "exp(" << input_value << ") = " << actual
            << ", log(actual) = " << inverted;
      });
}

TEST(AvxActivationFunctionsTest, SigmoidTest) {
  AvxVectorFuzzTest(  //
      [](AvxFloatVec *vec) { *vec = activations::Sigmoid(*vec); },
      [](float input_value, float actual) {
        const float expected = 1.0f / (1.0f + exp(-input_value));
        EXPECT_NEAR(actual, expected, 1e-6)
            << "sigmoid(" << input_value << ") = " << actual
            << ", expected = " << expected;
      });
}

template <int batch_size, class Function>
void RunPerformanceTest(Function activation, int flops) {
  constexpr uint64 kIterations = 1000000;

  UniqueVector<float> input(batch_size);
  UniqueVector<float> output(batch_size);
  InitRandomVector(*input);
  InitRandomVector(*output);

  AvxFloatVecArray<batch_size / kAvxWidth> array;
  auto start_time = std::chrono::system_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    array.Load(input->data());
    array.Apply(activation);
    array.Store(output->data());
  }
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  double elapsed = elapsed_seconds.count();
  double exp_ops = kIterations * batch_size;
  double macro_gops = exp_ops / 1e9 / elapsed;
  VLOG(0) << "For batch_size " << batch_size
          << " macro-GOPS (giga-ops per sec): " << macro_gops
          << ", raw arithmetic: " << flops * macro_gops;
}

TEST(AvxActivationFunctionsTest, SigmoidPerformanceTest) {
  RunPerformanceTest<8>(activations::Sigmoid, 26);
  RunPerformanceTest<16>(activations::Sigmoid, 26);
  RunPerformanceTest<32>(activations::Sigmoid, 26);
  RunPerformanceTest<48>(activations::Sigmoid, 26);
  RunPerformanceTest<64>(activations::Sigmoid, 26);
  RunPerformanceTest<128>(activations::Sigmoid, 26);
}

TEST(AvxActivationFunctionsTest, TanhTest) {
  AvxVectorFuzzTest([](AvxFloatVec *vec) { *vec = activations::Tanh(*vec); },
                    [](float input_value, float actual) {
                      const float expected = tanh(input_value);
                      EXPECT_NEAR(actual, expected, 1e-6)
                          << "tanh(" << input_value << ") = " << actual
                          << ", expected = " << expected;
                    });
}

TEST(AvxActivationFunctionsTest, TanhPerformanceTest) {
  RunPerformanceTest<8>(activations::Sigmoid, 23);
  RunPerformanceTest<16>(activations::Sigmoid, 23);
  RunPerformanceTest<32>(activations::Tanh, 23);
  RunPerformanceTest<48>(activations::Tanh, 23);
  RunPerformanceTest<64>(activations::Tanh, 23);
  RunPerformanceTest<128>(activations::Tanh, 23);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
