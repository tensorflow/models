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

#include "dragnn/runtime/lstm_network_kernel.h"

#include <vector>

#include "dragnn/runtime/attributes.h"
#include "dragnn/runtime/math/avx_activation_functions.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Attributes used by the LSTM network.
struct LSTMNetworkAttributes : public Attributes {
  // Hidden layer sizes; e.g. "96". LSTMNetwork only supports a single hidden
  // layer size.
  Mandatory<size_t> hidden_layer_sizes{"hidden_layer_sizes", this};

  // Whether to omit the "logits" layer.
  Optional<bool> omit_logits{"omit_logits", false, this};

  // Whether to use truncated floating-point weight matrices. This incurs very
  // large errors in the actual matrix multiplication, but the LSTM architecture
  // seems to be mostly resilient (99.99% similar performance on the tagger).
  Optional<bool> use_bfloat16_matrices{"use_bfloat16_matrices", false, this};

  // Training-only attributes, ignored in the runtime.
  Ignored dropout_keep_prob{"dropout_keep_prob", this};
  Ignored dropout_per_sequence{"dropout_per_sequence", this};
  Ignored dropout_all_layers{"dropout_all_layers", this};
  Ignored initialize_bias_zero{"initialize_bias_zero", this};
  Ignored initialize_softmax_zero{"initialize_softmax_zero", this};
  Ignored initialize_hidden_orthogonal{"initialize_hidden_orthogonal", this};
};

// Initalizes a LstmCellFunction, using the names that are emitted by
// network_units.py's LSTMNetwork class.
template <typename MatrixElementType>
tensorflow::Status InitializeLstmCellFunction(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    LstmCellFunction<MatrixElementType> *cell_function);

}  // namespace

string LSTMNetworkKernel::GetLogitsName() const {
  return has_logits_ ? FeedForwardNetworkLayer::kLogitsName : "";
}

tensorflow::Status LSTMNetworkKernel::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  // Parse network configuration.
  LSTMNetworkAttributes attributes;
  TF_RETURN_IF_ERROR(
      attributes.Reset(component_spec.network_unit().parameters()));
  has_logits_ = !TransitionSystemTraits(component_spec).is_deterministic &&
                !attributes.omit_logits();
  const int hidden_dimension = attributes.hidden_layer_sizes();

  // Initialize the LSTM cell.
  if (attributes.use_bfloat16_matrices()) {
    LstmCellFunction<TruncatedFloat16> *bfloat16_cell_function =
        new LstmCellFunction<TruncatedFloat16>();
    cell_function_.reset(bfloat16_cell_function);
    TF_RETURN_IF_ERROR(InitializeLstmCellFunction(
        component_spec, variable_store, bfloat16_cell_function));
  } else {
    LstmCellFunction<float> *float32_cell_function =
        new LstmCellFunction<float>();
    cell_function_.reset(float32_cell_function);
    TF_RETURN_IF_ERROR(InitializeLstmCellFunction(
        component_spec, variable_store, float32_cell_function));
  }

  // Add a softmax to compute logits, if necessary.
  if (has_logits_) {
    TF_RETURN_IF_ERROR(softmax_layer_.InitializeSoftmax(
        component_spec, variable_store, network_state_manager));
  }

  // Internal state layers.
  TF_RETURN_IF_ERROR(
      network_state_manager->AddLocal(hidden_dimension, &cell_state_));
  TF_RETURN_IF_ERROR(
      network_state_manager->AddLocal(hidden_dimension, &cell_output_));
  if (bulk_) {
    TF_RETURN_IF_ERROR(network_state_manager->AddLocal(
        3 * hidden_dimension, &cell_input_matrix_));
  } else {
    TF_RETURN_IF_ERROR(network_state_manager->AddLocal(
        3 * hidden_dimension, &cell_input_vector_));
  }

  // Layers exposed to the system.
  TF_RETURN_IF_ERROR(network_state_manager->AddLayer(
      "layer_0", hidden_dimension, &hidden_));
  TF_RETURN_IF_ERROR(
      network_state_manager->AddLayerAlias("last_layer", "layer_0"));

  return tensorflow::Status::OK();
}

tensorflow::Status LSTMNetworkKernel::Apply(size_t step_index,
                                            Vector<float> input,
                                            SessionState *session_state) const {
  DCHECK(!bulk_);
  const NetworkStates &network_states = session_state->network_states;
  const bool is_initial = step_index == 0;

  MutableVector<float> cell_state = network_states.GetLocal(cell_state_);
  MutableVector<float> cell_output = network_states.GetLocal(cell_output_);
  MutableMatrix<float> hidden_all_steps = network_states.GetLayer(hidden_);
  MutableVector<float> next_hidden = hidden_all_steps.row(step_index);

  // c_{t-1} and h_t vectors. These will be null if not applicable, so incorrect
  // code will immediately segfault.
  Vector<float> last_cell_state;
  Vector<float> last_hidden;
  if (!is_initial) {
    last_cell_state = cell_state;
    last_hidden = hidden_all_steps.row(step_index - 1);
  }

  // Run the cell function.
  MutableVector<float> cell_input = network_states.GetLocal(cell_input_vector_);
  TF_RETURN_IF_ERROR(cell_function_->Run(is_initial, input, last_hidden,
                                         last_cell_state, cell_input,
                                         cell_state, cell_output, next_hidden));

  // Compute logits, if present.
  if (has_logits_) {
    softmax_layer_.Apply(Vector<float>(next_hidden), network_states,
                         step_index);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status LSTMNetworkKernel::Apply(Matrix<float> all_inputs,
                                            SessionState *session_state) const {
  DCHECK(bulk_);
  const NetworkStates &network_states = session_state->network_states;
  const size_t num_steps = all_inputs.num_rows();

  MutableMatrix<float> all_cell_input_temps =
      network_states.GetLocal(cell_input_matrix_);
  MutableVector<float> cell_state = network_states.GetLocal(cell_state_);
  MutableVector<float> cell_output = network_states.GetLocal(cell_output_);
  MutableMatrix<float> all_hiddens = network_states.GetLayer(hidden_);

  // SGEMVV input computation.
  TF_RETURN_IF_ERROR(
      cell_function_->RunInputComputations(all_inputs, all_cell_input_temps));

  // Run recurrent parts of the network.
  for (size_t i = 0; i < num_steps; ++i) {
    const bool is_initial = i == 0;
    Vector<float> last_cell_state;
    Vector<float> last_hidden;
    if (!is_initial) {
      last_cell_state = cell_state;
      last_hidden = all_hiddens.row(i - 1);
    }
    TF_RETURN_IF_ERROR(cell_function_->RunRecurrentComputation(
        is_initial, last_hidden, last_cell_state, all_cell_input_temps.row(i),
        cell_state, cell_output, all_hiddens.row(i)));
  }

  if (has_logits_) {
    softmax_layer_.Apply(Matrix<float>(all_hiddens), network_states);
  }

  return tensorflow::Status::OK();
}

namespace {

// Returns a variable suffix for the |ElementType|.
template <typename ElementType>
string MatrixElementTypeSuffix();
template <>
string MatrixElementTypeSuffix<float>() {
  return "";
}
template <>
string MatrixElementTypeSuffix<TruncatedFloat16>() {
  return "/bfloat16";
}

// Shared logic for initializing SGEMV matrices.
template <int block_size, typename ElementType>
tensorflow::Status InitializeSgemv(
    const string &weights_name, VariableStore *variable_store,
    SgemvMatrix<block_size, ElementType> *sgemv_matrix) {
  BlockedMatrix<ElementType> blocked_transpose;
  TF_RETURN_IF_ERROR(variable_store->Lookup(
      tensorflow::strings::StrCat(weights_name, "/matrix/blocked", block_size,
                                  MatrixElementTypeSuffix<ElementType>()),
      &blocked_transpose));
  auto blocked = blocked_transpose.Transpose();
  auto result = sgemv_matrix->Initialize(blocked);
  if (result.ok()) {
    LOG(INFO) << "Matrix of size " << blocked.num_rows() << " x "
              << blocked.num_columns() << " for layer " << weights_name
              << " will be computed with SGEMV<block_size=" << block_size
              << ">";
  } else {
    // This should (almost?) never happen, because sgemv_matrix->Initialize()
    // only fails on bad block sizes, and we request the same block size from
    // the variable store.
    LOG(ERROR) << "Error formatting SGEMV matrix: " << result.error_message()
               << " - matrix size " << blocked.num_rows() << " x "
               << blocked.num_columns() << " for layer " << weights_name;
  }
  return result;
}

// Initalizes a LstmCellFunction, using the names that are emitted by
// network_units.py's LSTMNetwork class.
template <typename MatrixElementType>
tensorflow::Status InitializeLstmCellFunction(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    LstmCellFunction<MatrixElementType> *cell_function) {
  LSTMNetworkAttributes attributes;
  TF_RETURN_IF_ERROR(
      attributes.Reset(component_spec.network_unit().parameters()));
  constexpr int kBatchSize = LstmCellFunction<>::kBatchSize;
  int hidden_dimension = attributes.hidden_layer_sizes();

  auto get_sgemv = [&](const string &name_suffix,
                       SgemvMatrix<kBatchSize, MatrixElementType> *matrix) {
    string name =
        tensorflow::strings::StrCat(component_spec.name(), name_suffix);
    return InitializeSgemv(name, variable_store, matrix);
  };

  SgemvMatrix<kBatchSize, MatrixElementType> input_to_cell_input_state_output,
      last_hidden_to_cell_input_state_output, last_cell_state_to_cell_input,
      cell_state_to_cell_output;
  TF_RETURN_IF_ERROR(get_sgemv("/x_to_ico", &input_to_cell_input_state_output));
  TF_RETURN_IF_ERROR(
      get_sgemv("/h_to_ico", &last_hidden_to_cell_input_state_output));
  TF_RETURN_IF_ERROR(get_sgemv("/c2i", &last_cell_state_to_cell_input));
  TF_RETURN_IF_ERROR(get_sgemv("/c2o", &cell_state_to_cell_output));

  string ico_bias_name =
      tensorflow::strings::StrCat(component_spec.name(), "/", "ico_bias");
  Vector<float> ico_bias;
  TF_RETURN_IF_ERROR(variable_store->Lookup(ico_bias_name, &ico_bias));
  return cell_function->Initialize(
      hidden_dimension, ico_bias, input_to_cell_input_state_output,
      last_hidden_to_cell_input_state_output, last_cell_state_to_cell_input,
      cell_state_to_cell_output);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
