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

#include "dragnn/runtime/operands.h"

#include <string.h>
#include <tuple>
#include <utility>
#include <vector>

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Expects that the two pointers are the same.
void ExpectSameAddress(const void *pointer1, const void *pointer2) {
  EXPECT_EQ(pointer1, pointer2);
}

// Sets the |vector| to |size| copies of the |value|.
template <class T>
void Fill(MutableVector<T> vector, size_t size, T value) {
  ASSERT_EQ(vector.size(), size);
  for (T &element : vector) element = value;
}

// Expects that the |vector| contains |size| copies of the |expected_value|.
template <class T>
void ExpectFilled(Vector<T> vector, size_t size, T expected_value) {
  ASSERT_EQ(vector.size(), size);
  for (const T element : vector) EXPECT_EQ(element, expected_value);
}

// Tests that OperandManager can add operands and remember their configuration.
TEST(OperandManagerTest, Add) {
  OperandManager manager;
  const OperandHandle handle1 = manager.Add({OperandType::kSingular, 7});
  const OperandHandle handle2 = manager.Add({OperandType::kStepwise, 11});

  EXPECT_EQ(manager.spec(handle1).type, OperandType::kSingular);
  EXPECT_EQ(manager.spec(handle1).size, 7);
  EXPECT_EQ(manager.spec(handle2).type, OperandType::kStepwise);
  EXPECT_EQ(manager.spec(handle2).size, 11);
}

// Tests that Operands contains operands whose dimensions match its manager.
TEST(OperandsTest, Dimensions) {
  const size_t kDim1 = 3, kDim2 = 41, kDim3 = 19, kDim4 = 77;

  OperandManager manager;
  const OperandHandle handle1 =
      manager.Add({OperandType::kSingular, kDim1 * sizeof(float)});
  const OperandHandle handle2 =
      manager.Add({OperandType::kStepwise, kDim2 * sizeof(double)});
  const OperandHandle handle3 =
      manager.Add({OperandType::kSingular, kDim3 * sizeof(float)});
  const OperandHandle handle4 =
      manager.Add({OperandType::kStepwise, kDim4 * sizeof(int)});

  AlignedView view;
  AlignedArea area;
  Operands operands;
  operands.Reset(&manager, 10);

  view = operands.GetSingular(handle1);
  EXPECT_EQ(view.size(), kDim1 * sizeof(float));
  EXPECT_EQ(Vector<float>(view).size(), kDim1);

  area = operands.GetStepwise(handle2);
  EXPECT_EQ(area.num_views(), 0);  // no steps yet
  EXPECT_EQ(area.view_size(), kDim2 * sizeof(double));
  EXPECT_EQ(Matrix<double>(area).num_rows(), 0);  // starts with no steps
  EXPECT_EQ(Matrix<double>(area).num_columns(), kDim2);

  view = operands.GetSingular(handle3);
  EXPECT_EQ(view.size(), kDim3 * sizeof(float));
  EXPECT_EQ(Vector<float>(view).size(), kDim3);

  area = operands.GetStepwise(handle4);
  EXPECT_EQ(area.num_views(), 0);  // no steps yet
  EXPECT_EQ(area.view_size(), kDim4 * sizeof(int));
  EXPECT_EQ(Matrix<int>(area).num_rows(), 0);  // starts with no steps
  EXPECT_EQ(Matrix<int>(area).num_columns(), kDim4);
}

// Tests that Operands can incrementally extend stepwise operands while
// preserving existing values.
TEST(OperandsTest, AddStepToStepwise) {
  const size_t kDim1 = 23, kDim2 = 29;

  OperandManager manager;
  const OperandHandle handle1 =
      manager.Add({OperandType::kStepwise, kDim1 * sizeof(double)});
  const OperandHandle handle2 =
      manager.Add({OperandType::kStepwise, kDim2 * sizeof(int)});

  Operands operands;
  operands.Reset(&manager, 10);

  // Repeatedly add a step and fill it with values.
  for (int i = 0; i < 100; ++i) {
    operands.AddStep();
    Fill(MutableVector<double>(operands.GetStepwise(handle1).view(i)), kDim1,
         1000.0 + i);
    Fill(MutableVector<int>(operands.GetStepwise(handle2).view(i)), kDim2,
         2000 + i);
  }

  // Check that data from earlier steps is preserved across reallocations.
  for (int i = 0; i < 100; ++i) {
    ExpectFilled(Vector<double>(operands.GetStepwise(handle1).view(i)), kDim1,
                 1000.0 + i);
    ExpectFilled(Vector<int>(operands.GetStepwise(handle2).view(i)), kDim2,
                 2000 + i);
  }
}

// Tests that Operands can add multiple steps at once.
TEST(OperandsTest, AddStepsToStepwise) {
  const size_t kDim1 = 23, kDim2 = 29;

  OperandManager manager;
  const OperandHandle handle1 =
      manager.Add({OperandType::kStepwise, kDim1 * sizeof(double)});
  const OperandHandle handle2 =
      manager.Add({OperandType::kStepwise, kDim2 * sizeof(int)});

  Operands operands;
  operands.Reset(&manager, 10);

  // Repeatedly add blocks of steps and fill them with values.
  for (int i = 0; i < 100; ++i) {
    if (i % 10 == 0) operands.AddSteps(10);  // occasionally add a block
    Fill(MutableVector<double>(operands.GetStepwise(handle1).view(i)), kDim1,
         1000.0 + i);
    Fill(MutableVector<int>(operands.GetStepwise(handle2).view(i)), kDim2,
         2000 + i);
  }

  // Check that data from earlier steps is preserved across reallocations.
  for (int i = 0; i < 100; ++i) {
    ExpectFilled(Vector<double>(operands.GetStepwise(handle1).view(i)), kDim1,
                 1000.0 + i);
    ExpectFilled(Vector<int>(operands.GetStepwise(handle2).view(i)), kDim2,
                 2000 + i);
  }
}

// Tests that Operands can add multiple steps to a pairwise operand.
TEST(OperandsTest, AddStepsPairwise) {
  const size_t kDim1 = 4, kDim2 = 31;

  OperandManager manager;
  const OperandHandle handle1 = manager.Add({OperandType::kPairwise, kDim1});
  const OperandHandle handle2 = manager.Add({OperandType::kPairwise, kDim2});

  Operands operands;
  operands.Reset(&manager, 10);

  { // A 1x1 pairwise operand.
    operands.AddSteps(1);
    const MutableAlignedArea area1 = operands.GetPairwise(handle1);
    const MutableAlignedArea area2 = operands.GetPairwise(handle2);

    EXPECT_EQ(area1.num_views(), 1);
    EXPECT_EQ(area2.num_views(), 1);

    EXPECT_EQ(area1.view_size(), kDim1);
    EXPECT_EQ(area2.view_size(), kDim2);

    // Write to operands to test the validity of the underlying memory region.
    memset(area1.view(0).data(), 0, kDim1);
    memset(area2.view(0).data(), 0, kDim2);
  }

  { // A 10x10 pairwise operand.
    operands.AddSteps(9);
    const MutableAlignedArea area1 = operands.GetPairwise(handle1);
    const MutableAlignedArea area2 = operands.GetPairwise(handle2);

    EXPECT_EQ(area1.num_views(), 10);
    EXPECT_EQ(area2.num_views(), 10);

    EXPECT_EQ(area1.view_size(), 10 * kDim1);
    EXPECT_EQ(area2.view_size(), 10 * kDim2);

    // Infer the stride by comparing pointers between consecutive views.
    const size_t expected_stride =
        PadToAlignment(10 * kDim1) + PadToAlignment(10 * kDim2);
    EXPECT_EQ(area1.view(1).data() - area1.view(0).data(), expected_stride);
    EXPECT_EQ(area2.view(1).data() - area2.view(0).data(), expected_stride);

    // Write to operands to test the validity of the underlying memory region.
    memset(area1.view(9).data(), 0, 10 * kDim1);
    memset(area2.view(9).data(), 0, 10 * kDim2);
  }
}

// Tests that Operands can be reused by resetting them repeatedly, possibly
// switching between different managers.
TEST(OperandsTest, ResetWithDifferentManagers) {
  std::vector<OperandManager> managers;
  std::vector<std::tuple<OperandHandle, OperandHandle, OperandHandle>> handles;
  for (int dim = 0; dim < 10; ++dim) {
    managers.emplace_back();
    handles.emplace_back(
        managers.back().Add({OperandType::kSingular, dim * sizeof(double)}),
        managers.back().Add({OperandType::kStepwise, dim * sizeof(int)}),
        managers.back().Add({OperandType::kPairwise, dim * sizeof(float)}));
  }

  Operands operands;
  for (int trial = 0; trial < 10; ++trial) {
    for (int dim = 0; dim < 10; ++dim) {
      operands.Reset(&managers[dim], 10);
      const OperandHandle singular_handle = std::get<0>(handles[dim]);
      const OperandHandle stepwise_handle = std::get<1>(handles[dim]);
      const OperandHandle pairwise_handle = std::get<2>(handles[dim]);

      // Fill the singular operand.
      Fill(MutableVector<double>(operands.GetSingular(singular_handle)), dim,
           100.0 * trial + dim);

      // Check the singular operands.
      ExpectFilled(Vector<double>(operands.GetSingular(singular_handle)), dim,
                   100.0 * trial + dim);

      // Repeatedly add a step and fill it with values.
      for (int step = 0; step < 100; ++step) {
        operands.AddStep();
        Fill(MutableVector<int>(
                 operands.GetStepwise(stepwise_handle).view(step)),
             dim, 1000 * trial + 100 * dim + step);
      }

      // Check that data from earlier steps is preserved across reallocations.
      for (int step = 0; step < 100; ++step) {
        ExpectFilled(
            Vector<int>(operands.GetStepwise(stepwise_handle).view(step)), dim,
            1000 * trial + 100 * dim + step);
      }

      // Check the dimensions of pairwise operands.
      Matrix<float> pairwise(operands.GetPairwise(pairwise_handle));
      EXPECT_EQ(pairwise.num_rows(), 100);
      EXPECT_EQ(pairwise.num_columns(), 100 * dim);
    }
  }
}

// Tests that one OperandManager can be shared simultaneously between multiple
// Operands instances.
TEST(OperandsTest, SharedManager) {
  const size_t kDim = 17;

  OperandManager manager;
  const OperandHandle singular_handle =
      manager.Add({OperandType::kSingular, kDim * sizeof(double)});
  const OperandHandle stepwise_handle =
      manager.Add({OperandType::kStepwise, kDim * sizeof(int)});

  std::vector<Operands> operands_vec(10);
  for (Operands &operands : operands_vec) operands.Reset(&manager, 10);

  // Fill all singular operands.
  for (int trial = 0; trial < operands_vec.size(); ++trial) {
    const Operands &operands = operands_vec[trial];
    Fill(MutableVector<double>(operands.GetSingular(singular_handle)), kDim,
         3.0 * trial);
  }

  // Check all singular operands.
  for (int trial = 0; trial < operands_vec.size(); ++trial) {
    const Operands &operands = operands_vec[trial];
    ExpectFilled(Vector<double>(operands.GetSingular(singular_handle)), kDim,
                 3.0 * trial);
  }

  // Fill all stepwise operands.  Interleave operations on the operands on each
  // step, so all operands are "active" at the same time.
  for (int step = 0; step < 100; ++step) {
    for (int trial = 0; trial < 10; ++trial) {
      Operands &operands = operands_vec[trial];
      operands.AddStep();
      Fill(MutableVector<int>(operands.GetStepwise(stepwise_handle).view(step)),
           kDim, trial * 999 + step);
    }
  }

  // Check all stepwise operands.
  for (int step = 0; step < 100; ++step) {
    for (int trial = 0; trial < 10; ++trial) {
      const Operands &operands = operands_vec[trial];
      ExpectFilled(
          Vector<int>(operands.GetStepwise(stepwise_handle).view(step)), kDim,
          trial * 999 + step);
    }
  }
}

// Tests that an Operands uses all of the pre-allocated steps and reallocates
// exactly when it exhausts the pre-allocated array.
TEST(OperandsTest, UsesPreAllocatedSteps) {
  const size_t kBytes = 5;
  const size_t kPreAllocateNumSteps = 10;

  OperandManager manager;
  const OperandHandle handle = manager.Add({OperandType::kStepwise, kBytes});

  Operands operands;
  operands.Reset(&manager, kPreAllocateNumSteps);

  // The first N steps fit exactly in the pre-allocated array.  Access the base
  // of the stepwise array via the first view.
  operands.AddStep();
  char *const pre_allocated_data = operands.GetStepwise(handle).view(0).data();
  for (size_t step = 1; step < kPreAllocateNumSteps; ++step) {
    operands.AddStep();
    ASSERT_EQ(operands.GetStepwise(handle).view(0).data(), pre_allocated_data);
  }

  // The N+1'st step triggers a reallocation, which is guaranteed to yield a new
  // pointer because it creates a separate array and copies into it.
  operands.AddStep();
  ASSERT_NE(operands.GetStepwise(handle).view(0).data(), pre_allocated_data);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
