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

#include "dragnn/runtime/math/sgemvv.h"

#include <chrono>
#include <random>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/math/arithmetic.h"
#include "dragnn/runtime/math/transformations.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/test/helpers.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

void naive_sgemv(const MutableMatrix<float> &matrix, const float *v,
                 const float *b, float *y) {
  for (int row = 0; row < matrix.num_rows(); row++) {
    y[row] = b[row];
    for (int col = 0; col < matrix.num_columns(); col++) {
      y[row] += matrix.row(row)[col] * v[col];
    }
  }
}

// Everything except floats require copying.
template <class ElementType>
constexpr bool RequiresCopy();
template <>
constexpr bool RequiresCopy<TruncatedFloat16>() {
  return true;
}
#if defined(__F16C__)
template <>
constexpr bool RequiresCopy<IeeeFloat16>() {
  return true;
}
#endif
template <>
constexpr bool RequiresCopy<float>() {
  return false;
}

template <class ElementType>
void ConvertRow(Vector<float> input, MutableVector<ElementType> output);
template <>
void ConvertRow<float>(Vector<float> input, MutableVector<float> output) {}
template <>
void ConvertRow<TruncatedFloat16>(Vector<float> input,
                                  MutableVector<TruncatedFloat16> output) {
  CHECK_EQ(input.size() % 16, 0);
  CHECK_EQ(input.size(), output.size());

  for (int i = 0; i < input.size(); ++i) {
    int i_permuted = (i / 16) * 16 + FastUnpackPermutation(i % 16);
    output[i] = TruncatedFloat16::DebugFromFloat(input[i_permuted]);
  }
}
#if defined(__F16C__)
template <>
void ConvertRow<IeeeFloat16>(Vector<float> input,
                             MutableVector<IeeeFloat16> output) {
  CHECK_EQ(input.size() % 16, 0);
  CHECK_EQ(input.size(), output.size());
  for (int i = 0; i < input.size(); ++i) {
    output[i] = IeeeFloat16::DebugFromFloat(input[i]);
  }
}
#endif

// Converts a matrix to SGEMV. If the element type is not float, copies the
// matrix and then converts it.
template <int sse_batch_size, typename ElementType = float>
SgemvMatrix<sse_batch_size, ElementType> ConvertToSgemv(
    const Matrix<float> &matrix, UniqueMatrix<ElementType> *sgemv_storage) {
  MutableBlockedMatrix<ElementType, BlockedMatrixFormat::kRowBlockedColumnMajor>
      blocked;
  TF_EXPECT_OK(blocked.Reset(sgemv_storage->area(), matrix.num_rows(),
                             matrix.num_columns()));

  // TODO(googleuser): Clean this up when we can use C++17's `if constexpr`
  // ... then we will not have to introduce this raw pointer, which is either
  // an actual new variable or alias to `sgemv_storage`.
  UniqueMatrix<float> *uncompressed;
  if (RequiresCopy<ElementType>()) {
    uncompressed = new UniqueMatrix<float>((*sgemv_storage)->num_rows(),
                                           (*sgemv_storage)->num_columns());
  } else {
    // NOTE: Because we don't have C++17's `if constexpr`, we need to add a
    // reinterpret_cast, so this code can compile when ElementType != float.
    uncompressed = reinterpret_cast<UniqueMatrix<float> *>(sgemv_storage);
  }

  // Copy to the uncompressed matrix. If ElementType == float, this is just
  // the output, otherwise it's the temporary array.
  MutableBlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
      uncompressed_matrix;
  TF_EXPECT_OK(uncompressed_matrix.Reset(
      uncompressed->area(), matrix.num_rows(), matrix.num_columns()));
  TF_EXPECT_OK(CopyMatrix(matrix, &uncompressed_matrix));

  if (RequiresCopy<ElementType>()) {
    for (int i = 0; i < blocked.num_vectors(); ++i) {
      ConvertRow<ElementType>(Vector<float>(uncompressed_matrix.vector(i)),
                              blocked.vector(i));
    }
    delete uncompressed;
  }

  SgemvMatrix<sse_batch_size, ElementType> sgemv_matrix;
  TF_EXPECT_OK(sgemv_matrix.Initialize(blocked.AsConst()));
  return sgemv_matrix;
}

void InitRandomVector(MutableVector<float> vector) {
  // clock() is updated less frequently than a cycle counter, so keep around the
  // RNG just in case we initialize some vectors in less than a clock tick.
  static std::mt19937 *rng = new std::mt19937(clock());
  std::normal_distribution<float> distribution(0, 1);
  for (int i = 0; i < vector.size(); i++) {
    vector[i] = distribution(*rng);
  }
}

void InitRandomMatrix(MutableMatrix<float> matrix) {
  // See InitRandomVector comment.
  static std::mt19937 *rng = new std::mt19937(clock());
  std::normal_distribution<float> distribution(0, 1);
  GenerateMatrix(
      matrix.num_rows(), matrix.num_columns(),
      [&distribution](int row, int col) { return distribution(*rng); },
      &matrix);
}

TEST(SgemvvTest, MatmulNoBias) {
  constexpr int sse_batch_size = 32;
  constexpr int num_rows = 32;
  constexpr int num_columns = 15;
  constexpr int output_size = 8;

  constexpr int sgemv_views = num_rows * num_columns / sse_batch_size;
  static_assert(num_rows * num_columns % sse_batch_size == 0,
                "Bad matrix size");

  ASSERT_EQ(output_size % 8, 0) << "Output size must still be 32-byte aligned.";

  UniqueMatrix<float> matrix(num_rows, num_columns);
  UniqueMatrix<float> sgemv_storage(sgemv_views, sse_batch_size);
  UniqueVector<float> input_vector(num_columns);
  UniqueVector<float> output(num_rows);
  UniqueVector<float> expected(num_rows);

  // Random initialization for all variables/values.
  InitRandomMatrix(*matrix);
  InitRandomVector(*output);
  InitRandomVector(*expected);
  InitRandomVector(*input_vector);

  // Layout SGEMV matrix.
  SgemvMatrix<sse_batch_size> sgemv_matrix =
      ConvertToSgemv<sse_batch_size>(Matrix<float>(*matrix), &sgemv_storage);

  // SGEMV multiplication.
  SgemvInputBatch<1> inputs = {{input_vector->data()}, {nullptr}};
  SgemvOutputBatch<1> outputs = {{output->data()}};
  sgemv_matrix.MaskedMatrixMultiVectorProductNoInitial(inputs, output_size,
                                                       &outputs);

  // Naive algorithm.
  MultiplyMatrixAndVector<float>(Matrix<float>(*matrix),
                                 Vector<float>(*input_vector), *expected);

  // Check that results are equal.
  for (int i = 0; i < output_size; i++) {
    EXPECT_NEAR(output->data()[i], expected->data()[i], 1e-5);
  }
}

TEST(SgemvvTest, ErrorsWithBadMultiple) {
  // Pick num_rows which is (32-byte) alignable, but not a multiple of
  // sse_batch_size (32 floats). These should return errors.
  for (int num_rows = 8; num_rows < 32; num_rows += 8) {
    // Layout blocked matrix.
    UniqueMatrix<float> sgemv_storage(1, num_rows);
    MutableBlockedMatrix<float, BlockedMatrixFormat::kRowBlockedColumnMajor>
        blocked;
    TF_EXPECT_OK(blocked.Reset(sgemv_storage.area(), num_rows, 1));

    // Initialize SgemvvMatrix.
    SgemvMatrix<32> matrix;
    EXPECT_THAT(matrix.Initialize(blocked.AsConst()),
                test::IsErrorWithSubstr("must be equal to sse_batch_size"));
  }
}

template <typename ElementType>
string TypenameString();
template <>
string TypenameString<float>() {
  return "float32";
}
template <>
string TypenameString<TruncatedFloat16>() {
  return "bfloat16";
}
#if defined(__F16C__)
template <>
string TypenameString<IeeeFloat16>() {
  return "float16";
}
#endif

template <typename ElementType>
float ToleranceAt128();
template <>
float ToleranceAt128<float>() {
  return 1e-5;
}
template <>
float ToleranceAt128<TruncatedFloat16>() {
  return 1;
}
#if defined(__F16C__)
template <>
float ToleranceAt128<IeeeFloat16>() {
  return 1e-1;
}
#endif

template <int sse_batch_size, int num_rows, int num_cols, typename ElementType>
void RunPerformanceTest(int output_size) {
  constexpr int sgemv_views = num_rows * num_cols / sse_batch_size;
  static_assert(num_rows * num_cols % sse_batch_size == 0, "Bad matrix size");

  ASSERT_EQ(output_size % 8, 0) << "Output size must still be 32-byte aligned.";

  UniqueMatrix<float> matrix(num_rows, num_cols);
  UniqueMatrix<ElementType> sgemv_storage(sgemv_views, sse_batch_size);

  UniqueVector<float> initial_1(num_rows);
  UniqueVector<float> initial_2(num_rows);
  UniqueVector<float> vector_1(num_cols);
  UniqueVector<float> vector_2(num_cols);
  UniqueVector<float> output_1(num_rows);
  UniqueVector<float> output_2(num_rows);
  UniqueVector<float> expected_output_1(num_rows);
  UniqueVector<float> expected_output_2(num_rows);
  UniqueVector<float> untouched_output_1(num_rows);
  UniqueVector<float> untouched_output_2(num_rows);

  // Random initialization for all variables/values.
  InitRandomMatrix(*matrix);
  InitRandomVector(*initial_1);
  InitRandomVector(*initial_2);
  InitRandomVector(*output_1);
  InitRandomVector(*output_2);
  InitRandomVector(*expected_output_1);
  InitRandomVector(*expected_output_2);
  InitRandomVector(*vector_1);
  InitRandomVector(*vector_2);
  for (int i = 0; i < num_rows; i++) {
    (*untouched_output_1)[i] = (*output_1)[i];
    (*untouched_output_2)[i] = (*output_2)[i];
  }

  // Layout SGEMV matrix.
  SgemvMatrix<sse_batch_size, ElementType> sgemv_matrix =
      ConvertToSgemv<sse_batch_size, ElementType>(Matrix<float>(*matrix),
                                                  &sgemv_storage);

  naive_sgemv(*matrix, vector_1->data(), initial_1->data(),
              expected_output_1->data());
  naive_sgemv(*matrix, vector_2->data(), initial_2->data(),
              expected_output_2->data());

  double raw_flops_per_iteration = 2.0 * 2.0 * num_rows * num_cols;
  const uint64 iterations =
      static_cast<uint64>(std::round(4e9 / raw_flops_per_iteration));
  auto start_time = std::chrono::system_clock::now();
  SgemvInputBatch<2> inputs = {
      {vector_1->data(), vector_2->data()},
      {initial_1->data(), initial_2->data()},
  };
  SgemvOutputBatch<2> outputs = {{output_1->data(), output_2->data()}};
  if (num_rows == output_size) {
    for (int iter = 0; iter < iterations; iter++) {
      sgemv_matrix.template MatrixMultiVectorProduct<2, 0, 0>(inputs, &outputs);
    }
  } else {
    for (int iter = 0; iter < iterations; iter++) {
      sgemv_matrix.template MaskedMatrixMultiVectorProduct<2>(
          inputs, output_size, &outputs);
    }
  }
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time - start_time;
  double elapsed = elapsed_seconds.count();

  // Each MatrixVectorVectorProduct does 2 Matrix-vector ops, and each op does a
  // multiply and an add (2 floating-point operations) for each entry in the
  // matrix.
  string raw_gflops = "";
  if (num_rows != output_size) {
    raw_gflops = ::tensorflow::strings::StrCat(
        ", ", raw_flops_per_iteration * iterations / 1e9 / elapsed, " raw");
  }
  VLOG(0) << "    ElementType " << TypenameString<ElementType>() << " GFLOPS: "
          << (2.0 * 2.0 * output_size * num_cols * iterations) / 1e9 / elapsed
          << " effective" << raw_gflops;

  const float tolerance =
      ToleranceAt128<ElementType>() * (num_rows / 128.0) + 1e-5;
  for (int i = 0; i < output_size; i++) {
    EXPECT_NEAR(output_1->data()[i], expected_output_1->data()[i], tolerance);
    EXPECT_NEAR(output_2->data()[i], expected_output_2->data()[i], tolerance);
  }

  // Check that any non-output items are untouched.
  for (int i = output_size; i < num_rows; i++) {
    EXPECT_EQ((*output_1)[i], (*untouched_output_1)[i]);
    EXPECT_EQ((*output_2)[i], (*untouched_output_2)[i]);
  }
}

TEST(SgemvvTest, PerformanceAndAccuracyTest) {
  // Benchmarking is hard. Sometimes results vary between test runs, or are just
  // unreliable. This could be in part from CPU frequency scaling, and also how
  // favorably the memory allocator places data (coherence, etc.).
  constexpr int kNumBatches = 3;

  VLOG(0) << "64x64 32-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<32, 64, 64, float>(64);
#if defined(__F16C__)
    RunPerformanceTest<32, 64, 64, IeeeFloat16>(64);
#endif
  }

  VLOG(0) << "128x128 32-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<32, 128, 128, float>(128);
  }

  VLOG(0) << "256x256 32-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<32, 256, 256, float>(256);
#if defined(__F16C__)
    RunPerformanceTest<32, 256, 256, IeeeFloat16>(256);
#endif
    RunPerformanceTest<32, 256, 256, TruncatedFloat16>(256);
  }

  VLOG(0) << "96x96 48-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<48, 96, 96, float>(96);
  }

  VLOG(0) << "48x96 48-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<48, 48, 96, float>(48);
  }

  VLOG(0) << "40x96 48-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<48, 48, 96, float>(40);
  }

  // These larger matrices are about the same amount of computation as one
  // 96-dimensional LSTM cell (without output softmax).
  VLOG(0) << "480x96 48-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<48, 480, 96, float>(480);
#if defined(__F16C__)
    RunPerformanceTest<48, 480, 96, IeeeFloat16>(480);
#endif
    RunPerformanceTest<48, 480, 96, TruncatedFloat16>(480);
  }

  VLOG(0) << "472x96 48-batch-size test";
  for (int batch = 0; batch < kNumBatches; ++batch) {
    RunPerformanceTest<48, 480, 96, float>(472);
#if defined(__F16C__)
    RunPerformanceTest<48, 480, 96, IeeeFloat16>(472);
#endif
    RunPerformanceTest<48, 480, 96, TruncatedFloat16>(472);
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
