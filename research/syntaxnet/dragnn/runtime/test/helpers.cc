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

#include "dragnn/runtime/test/helpers.h"

#include <time.h>
#include <random>

#include "dragnn/runtime/math/transformations.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

UniqueView::UniqueView(size_t size) {
  array_.Reset(size);
  view_ = array_.view();
}

UniqueArea::UniqueArea(size_t num_views, size_t view_size) {
  array_.Reset(ComputeAlignedAreaSize(num_views, view_size));
  TF_CHECK_OK(area_.Reset(array_.view(), num_views, view_size));
}

void InitRandomVector(MutableVector<float> vector) {
  // clock() is updated less frequently than a cycle counter, so keep around the
  // RNG just in case we initialize some vectors in less than a clock tick.
  thread_local std::mt19937 *rng = new std::mt19937(clock());
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i = 0; i < vector.size(); i++) {
    vector[i] = distribution(*rng);
  }
}

void InitRandomMatrix(MutableMatrix<float> matrix) {
  // See InitRandomVector comment.
  thread_local std::mt19937 *rng = new std::mt19937(clock());
  std::normal_distribution<float> distribution(0.0, 1.0);
  GenerateMatrix(
      matrix.num_rows(), matrix.num_columns(),
      [&distribution](int row, int col) { return distribution(*rng); },
      &matrix);
}

void AvxVectorFuzzTest(
    const std::function<void(AvxFloatVec *vec)> &run,
    const std::function<void(float input_value, float output_value)> &check) {
  for (int iter = 0; iter < 100; ++iter) {
    UniqueVector<float> input(kAvxWidth);
    UniqueVector<float> output(kAvxWidth);
    InitRandomVector(*input);
    InitRandomVector(*output);

    AvxFloatVec vec;
    vec.Load(input->data());
    run(&vec);
    vec.Store(output->data());

    for (int i = 0; i < kAvxWidth; ++i) {
      check((*input)[i], (*output)[i]);
    }
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
