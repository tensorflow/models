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

// Utils for declaring and allocating operands.  An operand is made up of
// aligned byte arrays, and can be used as an input, output, or intermediate
// value in some computation.

#ifndef DRAGNN_RUNTIME_OPERANDS_H_
#define DRAGNN_RUNTIME_OPERANDS_H_

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/runtime/alignment.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Possible types of operands.
enum class OperandType {
  // A single byte array.  For example, an intermediate value that is computed
  // once per transition step.  Since it is not an output, the same storage
  // could be reused across all steps.
  kSingular,

  // A sequence of identically-sized byte arrays, one per transition step.  For
  // example, a layer containing one activation vector per step.
  kStepwise,

  // A grid with one byte array for each pair of transition steps, including
  // self pairings.  The byte arrays are grouped and concatenated in "rows",
  // forming one byte array per step.  For example, if there are N steps and D
  // bytes per pair, the operand would have N arrays of size N*D bytes.  In a
  // basic attention model with one "similarity" between pairs of steps, one
  // might use a pairwise operand with D=sizeof(float).  For best performance,
  // use Operands::AddSteps() to allocate all steps at once when working with
  // pairwise operands.
  kPairwise,
};

// A specification of a operand.
struct OperandSpec {
  // Creates a trivial specification.
  OperandSpec() = default;

  // Creates a specification with the |type| and |size|.
  OperandSpec(OperandType type, size_t size) : type(type), size(size) {}

  // Type of the operand.
  OperandType type = OperandType::kSingular;

  // Size of each aligned byte array in the operand.
  size_t size = 0;
};

// An opaque handle to an operand.
class OperandHandle;

// A class that manages a set of operand specifications and associates each
// operand with a handle.  Operand contents can be retrieved using these
// handles; see Operands below.
class OperandManager {
 public:
  // Creates an empty manager.
  OperandManager() = default;

  // Adds an operand configured according to the |spec| and returns its handle.
  OperandHandle Add(const OperandSpec &spec);

  // Accessors.
  const OperandSpec &spec(OperandHandle handle) const;

 private:
  friend class Operands;

  // Specification of each operand.
  std::vector<OperandSpec> specs_;

  // Mapping from the handle index of an operand to its index amongst operands
  // of the same type.
  std::vector<size_t> handle_index_to_typed_index_;

  // Span of each singular operand, as a (start-offset,size) pair, relative to
  // the byte array containing all singular operands.
  std::vector<std::pair<size_t, size_t>> singular_spans_;

  // Span of each stepwise operand, as a (start-offset,size) pair, relative to
  // the byte array for each step.
  std::vector<std::pair<size_t, size_t>> stepwise_spans_;

  // Size of each pairwise operand.
  std::vector<size_t> pairwise_sizes_;

  // Number of bytes used by all singular operands, including alignment padding.
  size_t singular_size_ = 0;

  // Number of bytes used by all stepwise operands on each step, including
  // alignment padding.
  size_t stepwise_stride_ = 0;
};

// A set of operands.  The structure of the operands is configured by an
// OperandManager, and operand contents can be accessed using the handles
// produced by the manager.
//
// Multiple Operands instances can share the same OperandManager.  In addition,
// an Operands instance can be reused by repeatedly Reset()-ing it, potentially
// with different OperandManagers.  Such reuse can reduce allocation overhead.
class Operands {
 public:
  // Creates an empty set.
  Operands() = default;

  // Resets this to the operands defined by the |manager|.  The |manager| must
  // live until this is destroyed or Reset() again, and should not be modified
  // during that time.  Stepwise and pairwise operands start with 0 steps; use
  // AddStep() to extend them.  Pre-allocates stepwise operands so that they
  // will not be reallocated during the first |pre_allocate_num_steps| calls to
  // AddStep().  Invalidates all previously-returned operands.
  void Reset(const OperandManager *manager, size_t pre_allocate_num_steps);

  // Extends stepwise and pairwise operands by one or more steps.  Requires that
  // Reset() was called.  Invalidates any previously-returned views of stepwise
  // and pairwise operands.  Preserves data for pre-existing steps of stepwise
  // operands, but not for pre-existing pairwise operands.  In general, pairwise
  // operands should be allocated in one shot, not incrementally.
  void AddStep() { AddSteps(1); }
  void AddSteps(size_t num_steps);

  // Returns the singular operand associated with the |handle|.  The returned
  // view is invalidated by Reset().
  MutableAlignedView GetSingular(OperandHandle handle) const;

  // Returns the stepwise operand associated with the |handle|.  The returned
  // area is invalidated by Reset() and AddStep().
  MutableAlignedArea GetStepwise(OperandHandle handle) const;

  // Returns the pairwise operand associated with the |handle|.  The returned
  // area is invalidated by Reset() and AddStep().
  MutableAlignedArea GetPairwise(OperandHandle handle) const;

 private:
  // Extends stepwise operands only; see AddSteps().
  void AddStepwiseSteps(size_t num_steps);

  // Extends pairwise operands only; see AddSteps().
  void AddPairwiseSteps(size_t num_steps);

  // Manager of the operands in this set.
  const OperandManager *manager_ = nullptr;

  // Cached members from the |manager_|.
  tensorflow::gtl::ArraySlice<size_t> handle_index_to_typed_index_;
  tensorflow::gtl::ArraySlice<std::pair<size_t, size_t>> stepwise_spans_;
  size_t stepwise_stride_ = 0;
  tensorflow::gtl::ArraySlice<size_t> pairwise_sizes_;

  // Byte arrays holding operands of each type.  Storage is separated because
  // each type grows differently with the number of steps.
  UniqueAlignedArray singular_array_;
  UniqueAlignedArray stepwise_array_;
  UniqueAlignedArray pairwise_array_;

  // Lists of operands of each type.
  std::vector<MutableAlignedView> singular_operands_;
  std::vector<MutableAlignedArea> stepwise_operands_;
  std::vector<MutableAlignedArea> pairwise_operands_;
};

// Implementation details below.

// An opaque handle to an operand.
class OperandHandle {
 public:
  // Creates an invalid handle.
  OperandHandle() = default;

 private:
  friend class OperandManager;
  friend class Operands;

  // Creates a handle that points to the |index|.
  explicit OperandHandle(size_t index) : index_(index) {}

  // Index of the operand in its manager.
  size_t index_ = SIZE_MAX;
};

inline const OperandSpec &OperandManager::spec(OperandHandle handle) const {
  return specs_[handle.index_];
}

inline MutableAlignedView Operands::GetSingular(OperandHandle handle) const {
  DCHECK(manager_->spec(handle).type == OperandType::kSingular)
      << "Actual type: " << static_cast<int>(manager_->spec(handle).type);
  DCHECK_LE(handle.index_, handle_index_to_typed_index_.size());
  return singular_operands_[handle_index_to_typed_index_[handle.index_]];
}

inline MutableAlignedArea Operands::GetStepwise(OperandHandle handle) const {
  DCHECK(manager_->spec(handle).type == OperandType::kStepwise)
      << "Actual type: " << static_cast<int>(manager_->spec(handle).type);
  DCHECK_LE(handle.index_, handle_index_to_typed_index_.size());
  return stepwise_operands_[handle_index_to_typed_index_[handle.index_]];
}

inline MutableAlignedArea Operands::GetPairwise(OperandHandle handle) const {
  DCHECK(manager_->spec(handle).type == OperandType::kPairwise)
      << "Actual type: " << static_cast<int>(manager_->spec(handle).type);
  DCHECK_LE(handle.index_, handle_index_to_typed_index_.size());
  return pairwise_operands_[handle_index_to_typed_index_[handle.index_]];
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_OPERANDS_H_
