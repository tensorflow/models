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

#include "dragnn/runtime/myelin/myelin_tracing.h"

#include <map>
#include <string>

#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Copies |num_values| |T|s from |data| into the |tensor_trace|.  If |T| does
// not match the |type|, returns false and modifies nothing.  The bool return
// allows this function to be chained until a matching type is found.
template <class T>
bool TryCopyValues(sling::myelin::Type type, const char *data, int num_values,
                   CellTensorTrace *tensor_trace) {
  if (sling::myelin::Traits<T>().type() != type) return false;
  const T *begin = reinterpret_cast<const T *>(data);
  const T *end = begin + num_values;
  tensor_trace->clear_value();
  for (; begin != end; ++begin) tensor_trace->add_value(*begin);
  return true;
}

}  // namespace

void TraceMyelinInstance(sling::myelin::Instance *instance,
                         CellTrace *cell_trace) {
  const sling::myelin::Cell &cell = *instance->cell();
  cell_trace->Clear();
  cell_trace->set_name(cell.name());

  // Collect steps and tensors in sorted maps for deterministic ordering.
  std::map<string, const sling::myelin::Step *> steps;
  std::map<string, sling::myelin::Tensor *> tensors;
  for (const sling::myelin::Step *step : cell.steps()) {
    steps[step->name()] = step;
    for (sling::myelin::Tensor *tensor : step->inputs()) {
      tensors[tensor->name()] = tensor;
    }
    for (sling::myelin::Tensor *tensor : step->outputs()) {
      tensors[tensor->name()] = tensor;
    }
  }

  // Trace each step as an operation.
  for (const auto &it : steps) {
    const sling::myelin::Step *step = it.second;
    CellOperationTrace *operation_trace = cell_trace->add_operation();
    operation_trace->set_name(step->name());
    operation_trace->set_type(step->type());
    operation_trace->set_kernel(step->kernel()->Name());
    for (sling::myelin::Tensor *tensor : step->inputs()) {
      operation_trace->add_input(tensor->name());
    }
    for (sling::myelin::Tensor *tensor : step->outputs()) {
      operation_trace->add_output(tensor->name());
    }
  }

  // Trace each tensor and its value.
  for (const auto &it : tensors) {
    sling::myelin::Tensor *tensor = it.second;
    if (!tensor->IsLocal()) continue;  // ignore globals; e.g., weight matrices
    const string &name = tensor->name();
    const sling::myelin::Type type = tensor->type();

    // Find the variable data for the |tensor|.  Note that ref tensors need to
    // be dereferenced.
    const char *data = instance->GetAddress(tensor);
    if (tensor->ref()) data = *reinterpret_cast<const char *const *>(data);
    const int size = tensor->aligned().elements();

    CellTensorTrace *tensor_trace = cell_trace->add_tensor();
    tensor_trace->set_name(name);
    tensor_trace->set_type(sling::myelin::TypeTraits::of(type).name());
    for (int i = 0; i < tensor->rank(); ++i) {
      tensor_trace->add_dimension(tensor->dim(i));
      tensor_trace->add_aligned_dimension(tensor->aligned(i));
    }

    switch (tensor->order()) {
      case sling::myelin::ROW_MAJOR:
        tensor_trace->set_order(CellTensorTrace::ORDER_ROW_MAJOR);
        break;

      case sling::myelin::COLUMN_MAJOR:
        tensor_trace->set_order(CellTensorTrace::ORDER_COLUMN_MAJOR);
        break;

      default:
        break;
    }

    // Try copying tensor data using all relevant types.  At most one attempt
    // will succeed and modify the |tensor_trace|.
    if (!TryCopyValues<float>(type, data, size, tensor_trace) &&
        !TryCopyValues<double>(type, data, size, tensor_trace) &&
        !TryCopyValues<bool>(type, data, size, tensor_trace) &&
        !TryCopyValues<int8>(type, data, size, tensor_trace) &&
        !TryCopyValues<int16>(type, data, size, tensor_trace) &&
        !TryCopyValues<int32>(type, data, size, tensor_trace) &&
        !TryCopyValues<int64>(type, data, size, tensor_trace) &&
        !TryCopyValues<uint8>(type, data, size, tensor_trace) &&
        !TryCopyValues<uint16>(type, data, size, tensor_trace)) {
      LOG(WARNING) << "Can't convert data for tensor " << name << " with type "
                   << tensor_trace->type();
    }
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
