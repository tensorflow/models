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

namespace syntaxnet {
namespace dragnn {
namespace runtime {

OperandHandle OperandManager::Add(const OperandSpec &spec) {
  const size_t index = specs_.size();
  specs_.push_back(spec);

  switch (spec.type) {
    case OperandType::kSingular:
      handle_index_to_typed_index_.push_back(singular_spans_.size());
      singular_spans_.emplace_back(singular_size_, spec.size);
      singular_size_ += PadToAlignment(spec.size);
      break;

    case OperandType::kStepwise:
      handle_index_to_typed_index_.push_back(stepwise_spans_.size());
      stepwise_spans_.emplace_back(stepwise_stride_, spec.size);
      stepwise_stride_ += PadToAlignment(spec.size);
      break;

    case OperandType::kPairwise:
      handle_index_to_typed_index_.push_back(pairwise_sizes_.size());
      pairwise_sizes_.push_back(spec.size);
      break;
  }

  return OperandHandle(index);
}

void Operands::Reset(const OperandManager *manager,
                     size_t pre_allocate_num_steps) {
  manager_ = manager;
  handle_index_to_typed_index_ = manager_->handle_index_to_typed_index_;
  stepwise_spans_ = manager_->stepwise_spans_;
  stepwise_stride_ = manager_->stepwise_stride_;
  pairwise_sizes_ = manager_->pairwise_sizes_;

  // Allocate and parcel out singular operands.
  singular_operands_.clear();
  singular_operands_.reserve(manager_->singular_spans_.size());
  singular_array_.Reserve(manager_->singular_size_);
  char *data = singular_array_.view().data();
  for (const auto &span : manager_->singular_spans_) {
    singular_operands_.push_back(
        MutableAlignedView(data + span.first, span.second));
  }

  // Pre-allocate and parcel out stepwise operands.
  stepwise_operands_.clear();
  stepwise_operands_.reserve(stepwise_spans_.size());
  stepwise_array_.Reserve(stepwise_stride_ * pre_allocate_num_steps);
  data = stepwise_array_.view().data();
  for (const auto &span : stepwise_spans_) {
    stepwise_operands_.push_back(MutableAlignedArea(
        data + span.first, 0, span.second, stepwise_stride_));
  }

  // Create empty pairwise operands.
  pairwise_operands_.clear();
  pairwise_operands_.resize(pairwise_sizes_.size());
}

void Operands::AddSteps(size_t num_steps) {
  AddStepwiseSteps(num_steps);
  AddPairwiseSteps(num_steps);
}

void Operands::AddStepwiseSteps(size_t num_steps) {
  if (stepwise_operands_.empty()) return;

  // Make room for the new steps.
  const size_t new_num_views = stepwise_operands_[0].num_views_ + num_steps;
  const bool actually_reallocated =
      stepwise_array_.Resize(new_num_views * stepwise_stride_);

  // Update the base pointers for stepwise operands, if changed.
  if (actually_reallocated) {
    char *data = stepwise_array_.view().data();
    for (size_t i = 0; i < stepwise_operands_.size(); ++i) {
      stepwise_operands_[i].data_ = data + stepwise_spans_[i].first;
    }
  }

  // Update the number of views in each stepwise operand.
  for (MutableAlignedArea &operand : stepwise_operands_) {
    operand.num_views_ = new_num_views;
  }
}

void Operands::AddPairwiseSteps(size_t num_steps) {
  if (pairwise_operands_.empty()) return;

  const size_t new_num_steps = pairwise_operands_[0].num_views_ + num_steps;

  // Set dimensions for each pairwise operand and accumulate their total stride.
  size_t new_stride = 0;
  for (size_t i = 0; i < pairwise_operands_.size(); ++i) {
    const size_t new_view_size = new_num_steps * pairwise_sizes_[i];
    pairwise_operands_[i].num_views_ = new_num_steps;
    pairwise_operands_[i].view_size_ = new_view_size;
    new_stride += PadToAlignment(new_view_size);
  }

  // Note that Reset() does not preserve the existing array and its contents.
  // Although preserving existing data would be nice, it is complex because
  // pairwise operands grow in both dimensions.  In addition, users should be
  // allocating pairwise operands in one shot for speed reasons, in which case
  // there is no existing data anyways.
  pairwise_array_.Reset(new_num_steps * new_stride);

  // Set the new base pointer and stride on each pairwise operand.
  char *data = pairwise_array_.view().data();
  for (MutableAlignedArea &operand : pairwise_operands_) {
    operand.data_ = data;
    operand.view_stride_ = new_stride;
    data += PadToAlignment(operand.view_size_);
  }

  DCHECK_EQ(data - pairwise_array_.view().data(), new_stride);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
