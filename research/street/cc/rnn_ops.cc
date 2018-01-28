/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// OpKernel of LSTM Neural Networks:
//
//   LSTM: VariableLSTMOp (VariableLSTMGradOp)
//
// where (.*) are the ops to compute gradients for the corresponding ops.

#define EIGEN_USE_THREADS

#include <vector>
#ifdef GOOGLE_INCLUDES
#include "third_party/eigen3/Eigen/Core"
#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#else
#include "Eigen/Core"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#endif  // GOOGLE_INCLUDES

namespace tensorflow {

using Eigen::array;
using Eigen::DenseIndex;
using IndexPair = Eigen::IndexPair<int>;

Status AreDimsEqual(int dim1, int dim2, const string& message) {
  if (dim1 != dim2) {
    return errors::InvalidArgument(message, ": ", dim1, " vs. ", dim2);
  }
  return Status::OK();
}

// ------------------------------- VariableLSTMOp -----------------------------

// Kernel to compute the forward propagation of a Long Short-Term Memory
// network. See the doc of the op below for more detail.
class VariableLSTMOp : public OpKernel {
 public:
  explicit VariableLSTMOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("clip", &clip_));
    OP_REQUIRES(
        ctx, clip_ >= 0.0,
        errors::InvalidArgument("clip_ needs to be equal or greator than 0"));
  }

  void Compute(OpKernelContext* ctx) override {
    // Inputs.
    const auto input = ctx->input(0).tensor<float, 4>();
    const auto initial_state = ctx->input(1).tensor<float, 2>();
    const auto initial_memory = ctx->input(2).tensor<float, 2>();
    const auto w_m_m = ctx->input(3).tensor<float, 3>();
    const int batch_size = input.dimension(0);
    const int seq_len = input.dimension(1);
    const int output_dim = input.dimension(3);

    // Sanity checks.
    OP_REQUIRES_OK(ctx, AreDimsEqual(4, input.dimension(2), "Input num"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, initial_state.dimension(0),
                                     "State batch"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(output_dim, initial_state.dimension(1), "State dim"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, initial_memory.dimension(0),
                                     "Memory batch"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(output_dim, initial_memory.dimension(1),
                                     "Memory dim"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(output_dim, w_m_m.dimension(0), "Weight dim 0"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(4, w_m_m.dimension(1), "Weight dim 1"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(output_dim, w_m_m.dimension(2), "Weight dim 2"));

    // Outputs.
    Tensor* act_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, {batch_size, seq_len, output_dim}, &act_tensor));
    auto act = act_tensor->tensor<float, 3>();
    act.setZero();

    Tensor* gate_raw_act_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, {batch_size, seq_len, 4, output_dim},
                                        &gate_raw_act_tensor));
    auto gate_raw_act = gate_raw_act_tensor->tensor<float, 4>();
    gate_raw_act.setZero();

    Tensor* memory_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, {batch_size, seq_len, output_dim},
                                        &memory_tensor));
    auto memory = memory_tensor->tensor<float, 3>();
    memory.setZero();

    // Const and scratch tensors.
    Tensor ones_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, {batch_size, output_dim},
                                           &ones_tensor));
    auto ones = ones_tensor.tensor<float, 2>();
    ones.setConstant(1.0);

    Tensor state_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, {batch_size, output_dim},
                                           &state_tensor));
    auto state = state_tensor.tensor<float, 2>();
    state = initial_state;

    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_FLOAT, {batch_size, 4, output_dim},
                                      &scratch_tensor));
    auto scratch = scratch_tensor.tensor<float, 3>();
    scratch.setZero();

    // Uses the most efficient order for the contraction depending on the batch
    // size.

    // This is the code shared by both cases. It is discouraged to use the
    // implicit capture with lambda functions, but it should be clear that what
    // is done here.
    auto Forward = [&](int i) {
      // Each pre-activation value is stored in the following order (See the
      // comment of the op for the meaning):
      //
      //   i: 0
      //   j: 1
      //   f: 2
      //   o: 3

      // Adds one to the pre-activation values of the forget gate. This is a
      // heuristic to make the training easier.
      scratch.chip(2, 1) += ones;

      gate_raw_act.chip(i, 1) = scratch;

      // c_t = f_t * c_{t-1} + i_t * j_t
      if (i == 0) {
        state = initial_memory * scratch.chip(2, 1).sigmoid();
      } else {
        state = memory.chip(i - 1, 1) * scratch.chip(2, 1).sigmoid();
      }
      state += scratch.chip(0, 1).sigmoid() * scratch.chip(1, 1).tanh();

      if (clip_ > 0.0) {
        // Clips the values if required.
        state = state.cwiseMax(-clip_).cwiseMin(clip_);
      }

      memory.chip(i, 1) = state;

      // h_t = o_t * tanh(c_t)
      state = scratch.chip(3, 1).sigmoid() * state.tanh();

      act.chip(i, 1) = state;
    };
    if (batch_size == 1) {
      // Reshapes the weight tensor to pretend as if it is a matrix
      // multiplication which is more efficient.
      auto w_m_m_r =
          w_m_m.reshape(array<DenseIndex, 2>{output_dim, 4 * output_dim});
      // Dimensions for the contraction.
      const array<IndexPair, 1> m_m_dim = {IndexPair(1, 0)};
      for (int i = 0; i < seq_len; ++i) {
        // Computes the pre-activation value of the input and each gate.
        scratch = input.chip(i, 1) +
                  state.contract(w_m_m_r, m_m_dim)
                      .reshape(array<DenseIndex, 3>{batch_size, 4, output_dim});
        Forward(i);
      }
    } else {
      // Shuffles the dimensions of the weight tensor to be efficient when used
      // in the left-hand side. Allocates memory for the shuffled tensor for
      // efficiency.
      Tensor w_m_m_s_tensor;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DT_FLOAT, {output_dim * 4, output_dim},
                                        &w_m_m_s_tensor));
      auto w_m_m_s = w_m_m_s_tensor.tensor<float, 2>();
      w_m_m_s = w_m_m.shuffle(array<int, 3>{2, 1, 0})
                    .reshape(array<DenseIndex, 2>{output_dim * 4, output_dim});
      // Dimensions for the contraction.
      const array<IndexPair, 1> m_m_dim = {IndexPair(1, 1)};
      for (int i = 0; i < seq_len; ++i) {
        // Computes the pre-activation value of the input and each gate.
        scratch = input.chip(i, 1) +
                  w_m_m_s.contract(state, m_m_dim)
                      .reshape(array<DenseIndex, 3>{output_dim, 4, batch_size})
                      .shuffle(array<int, 3>{2, 1, 0});
        Forward(i);
      }
    }
  }

 private:
  // Threshold to clip the values of memory cells.
  float clip_ = 0;
};

REGISTER_KERNEL_BUILDER(Name("VariableLSTM").Device(DEVICE_CPU),
                        VariableLSTMOp);
REGISTER_OP("VariableLSTM")
    .Attr("clip: float = 0.0")
    .Input("input: float32")
    .Input("initial_state: float32")
    .Input("initial_memory: float32")
    .Input("w_m_m: float32")
    .Output("activation: float32")
    .Output("gate_raw_act: float32")
    .Output("memory: float32")
    .Doc(R"doc(
Computes the forward propagation of a Long Short-Term Memory Network.

It computes the following equation recursively for `0<t<=T`:

  i_t  = sigmoid(a_{i,t})
  j_t  = tanh(a_{j,t})
  f_t  = sigmoid(a_{f,t} + 1.0)
  o_t  = sigmoid(a_{o,t})
  c_t  = f_t * c_{t-1} + i_t * j_t
  c'_t = min(max(c_t, -clip), clip) if clip > 0 else c_t
  h_t  = o_t * tanh(c'_t)

where

  a_{l,t} = w_{l,m,m} * h_{t-1} + x'_{l,t}

where

  x'_{l,t} = w_{l,m,i} * x_{t}.

`input` corresponds to the concatenation of `X'_i`, `X'_j`, `X'_f`, and `X'_o`
where `X'_l = (x'_{l,1}, x'_{l,2}, ..., x'_{l,T})`, `initial_state` corresponds
to `h_{0}`, `initial_memory` corresponds to `c_{0}` and `weight` corresponds to
`w_{l,m,m}`. `X'_l` (the transformed input) is computed outside of the op in
advance, so w_{l,m,i} is not passed in to the op.

`activation` corresponds to `H = (h_1, h_2, ..., h_T)`, `gate_raw_activation`
corresponds to the concatanation of `A_i`, `A_j`, `A_f` and `A_o`, and `memory`
corresponds `C = (c_0, c_1, ..., c_T)`.

All entries in the batch are propagated to the end, and are assumed to be the
same length.

input: 4-D with shape `[batch_size, seq_len, 4, num_nodes]`
initial_state: 2-D with shape `[batch_size, num_nodes]`
initial_memory: 2-D with shape `[batch_size, num_nodes]`
w_m_m: 3-D with shape `[num_nodes, 4, num_nodes]`
activation: 3-D with shape `[batch_size, seq_len, num_nodes]`
gate_raw_act: 3-D with shape `[batch_size, seq_len, 4, num_nodes]`
memory: 3-D with shape `[batch_size, seq_len, num_nodes]`
)doc");

// ----------------------------- VariableLSTMGradOp ----------------------------

// Kernel to compute the gradient of VariableLSTMOp.
class VariableLSTMGradOp : public OpKernel {
 public:
  explicit VariableLSTMGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Inputs.
    const auto initial_state = ctx->input(0).tensor<float, 2>();
    const auto initial_memory = ctx->input(1).tensor<float, 2>();
    const auto w_m_m = ctx->input(2).tensor<float, 3>();
    const auto act = ctx->input(3).tensor<float, 3>();
    const auto gate_raw_act = ctx->input(4).tensor<float, 4>();
    const auto memory = ctx->input(5).tensor<float, 3>();
    const auto act_grad = ctx->input(6).tensor<float, 3>();
    const auto gate_raw_act_grad = ctx->input(7).tensor<float, 4>();
    const auto memory_grad = ctx->input(8).tensor<float, 3>();
    const int batch_size = act.dimension(0);
    const int seq_len = act.dimension(1);
    const int output_dim = act.dimension(2);

    // Sanity checks.
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, initial_state.dimension(0),
                                     "State batch"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(output_dim, initial_state.dimension(1), "State dim"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, initial_memory.dimension(0),
                                     "Memory batch"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(output_dim, initial_memory.dimension(1),
                                     "Memory dim"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(output_dim, w_m_m.dimension(0), "Weight dim 0"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(4, w_m_m.dimension(1), "Weight dim 1"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(output_dim, w_m_m.dimension(2), "Weight dim 2"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, gate_raw_act.dimension(0),
                                     "Gate raw activation batch"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(seq_len, gate_raw_act.dimension(1),
                                     "Gate raw activation  len"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(4, gate_raw_act.dimension(2),
                                     "Gate raw activation num"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(output_dim, gate_raw_act.dimension(3),
                                     "Gate raw activation dim"));
    OP_REQUIRES_OK(
        ctx, AreDimsEqual(batch_size, memory.dimension(0), "Memory batch"));
    OP_REQUIRES_OK(ctx,
                   AreDimsEqual(seq_len, memory.dimension(1), "Memory len"));
    OP_REQUIRES_OK(ctx,
                   AreDimsEqual(output_dim, memory.dimension(2), "Memory dim"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, act_grad.dimension(0),
                                     "Activation gradient batch"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(seq_len, act_grad.dimension(1),
                                     "Activation gradient len"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(output_dim, act_grad.dimension(2),
                                     "Activation gradient dim"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, gate_raw_act_grad.dimension(0),
                                     "Activation gradient batch"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(seq_len, gate_raw_act_grad.dimension(1),
                                     "Activation gradient len"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(4, gate_raw_act_grad.dimension(2),
                                     "Activation gradient num"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(output_dim, gate_raw_act_grad.dimension(3),
                                     "Activation gradient dim"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(batch_size, memory_grad.dimension(0),
                                     "Memory gradient batch"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(seq_len, memory_grad.dimension(1),
                                     "Memory gradient len"));
    OP_REQUIRES_OK(ctx, AreDimsEqual(output_dim, memory_grad.dimension(2),
                                     "Memory gradient dim"));

    // Outputs.
    std::vector<Tensor*> collections(4, nullptr);
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, {batch_size, seq_len, 4, output_dim},
                                        &collections[0]));
    auto input_grad = collections[0]->tensor<float, 4>();
    input_grad.setZero();

    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {batch_size, output_dim},
                                             &collections[1]));
    auto init_state_grad = collections[1]->tensor<float, 2>();
    init_state_grad.setZero();

    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {batch_size, output_dim},
                                             &collections[2]));
    auto init_memory_grad = collections[2]->tensor<float, 2>();
    init_memory_grad.setZero();

    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, {output_dim, 4, output_dim},
                                             &collections[3]));
    auto w_m_m_grad = collections[3]->tensor<float, 3>();
    w_m_m_grad.setZero();

    // Const and scratch tensors.
    Tensor ones_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, {batch_size, output_dim},
                                           &ones_tensor));
    auto ones = ones_tensor.tensor<float, 2>();
    ones.setConstant(1.0);

    Tensor scratch_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_FLOAT, {batch_size, 4, output_dim},
                                      &scratch_tensor));
    auto scratch = scratch_tensor.tensor<float, 3>();
    scratch.setZero();

    Tensor tmp1_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, {batch_size, output_dim},
                                           &tmp1_tensor));
    auto tmp1 = tmp1_tensor.tensor<float, 2>();
    tmp1.setZero();

    Tensor tmp2_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, {batch_size, output_dim},
                                           &tmp2_tensor));
    auto tmp2 = tmp2_tensor.tensor<float, 2>();
    tmp2.setZero();

    // Uses the most efficient order for the contraction depending on the batch
    // size.

    // Shuffles the dimensions of the weight tensor to be efficient when used in
    // the left-hand side. Allocates memory for the shuffled tensor for
    // efficiency.
    Tensor w_m_m_s_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_FLOAT, {4, output_dim, output_dim},
                                      &w_m_m_s_tensor));
    auto w_m_m_s = w_m_m_s_tensor.tensor<float, 3>();
    if (batch_size == 1) {
      // Allocates memory only it is used.
      w_m_m_s = w_m_m.shuffle(array<int, 3>{1, 2, 0});
    }

    // Dimensions for the contraction with the weight tensor.
    const array<IndexPair, 1> m_m_dim =
        batch_size == 1 ? array<IndexPair, 1>{IndexPair(1, 0)}
                        : array<IndexPair, 1>{IndexPair(1, 1)};
    // Dimensions for the contraction of the batch dimensions.
    const array<IndexPair, 1> b_b_dim = {IndexPair(0, 0)};
    for (int i = seq_len - 1; i >= 0; --i) {
      if (i == seq_len - 1) {
        init_state_grad = act_grad.chip(i, 1);
      } else {
        w_m_m_grad +=
            act.chip(i, 1)
                .contract(scratch.reshape(
                              array<DenseIndex, 2>{batch_size, 4 * output_dim}),
                          b_b_dim)
                .reshape(array<DenseIndex, 3>{output_dim, 4, output_dim});
        if (batch_size == 1) {
          init_state_grad.device(ctx->eigen_cpu_device()) =
              scratch.chip(0, 1).contract(w_m_m_s.chip(0, 0), m_m_dim) +
              scratch.chip(1, 1).contract(w_m_m_s.chip(1, 0), m_m_dim) +
              scratch.chip(2, 1).contract(w_m_m_s.chip(2, 0), m_m_dim) +
              scratch.chip(3, 1).contract(w_m_m_s.chip(3, 0), m_m_dim);
        } else {
          init_state_grad.device(ctx->eigen_cpu_device()) =
              (w_m_m.chip(0, 1).contract(scratch.chip(0, 1), m_m_dim) +
               w_m_m.chip(1, 1).contract(scratch.chip(1, 1), m_m_dim) +
               w_m_m.chip(2, 1).contract(scratch.chip(2, 1), m_m_dim) +
               w_m_m.chip(3, 1).contract(scratch.chip(3, 1), m_m_dim))
                  .shuffle(array<int, 2>{1, 0});
        }
        init_state_grad += act_grad.chip(i, 1);
      }

      auto gate_raw_act_t = gate_raw_act.chip(i, 1);
      auto gate_raw_act_grad_t = gate_raw_act_grad.chip(i, 1);

      // Output gate.
      tmp1 = memory.chip(i, 1);
      tmp1 = tmp1.tanh();                          // y_t
      tmp2 = gate_raw_act_t.chip(3, 1).sigmoid();  // o_t
      scratch.chip(3, 1) = init_state_grad * tmp1 * tmp2 * (ones - tmp2) +
                           gate_raw_act_grad_t.chip(3, 1);

      init_memory_grad += init_state_grad * tmp2 * (ones - tmp1.square()) +
                          memory_grad.chip(i, 1);

      // Input gate.
      tmp1 = gate_raw_act_t.chip(0, 1).sigmoid();  // i_t
      tmp2 = gate_raw_act_t.chip(1, 1);
      tmp2 = tmp2.tanh();  // j_t
      scratch.chip(0, 1) = init_memory_grad * tmp2 * tmp1 * (ones - tmp1) +
                           gate_raw_act_grad_t.chip(0, 1);

      // Input.
      scratch.chip(1, 1) = init_memory_grad * tmp1 * (ones - tmp2.square()) +
                           gate_raw_act_grad_t.chip(1, 1);

      // Forget gate.
      tmp1 = gate_raw_act_t.chip(2, 1).sigmoid();  // f_t
      if (i == 0) {
        scratch.chip(2, 1) =
            init_memory_grad * initial_memory * tmp1 * (ones - tmp1) +
            gate_raw_act_grad_t.chip(2, 1);
      } else {
        scratch.chip(2, 1) =
            init_memory_grad * memory.chip(i - 1, 1) * tmp1 * (ones - tmp1) +
            gate_raw_act_grad_t.chip(2, 1);
      }

      // Memory.
      init_memory_grad *= tmp1;

      input_grad.chip(i, 1) = scratch;
    }
    w_m_m_grad += initial_state
                      .contract(scratch.reshape(array<DenseIndex, 2>{
                                    batch_size, 4 * output_dim}),
                                b_b_dim)
                      .reshape(array<DenseIndex, 3>{output_dim, 4, output_dim});
    if (batch_size == 1) {
      init_state_grad.device(ctx->eigen_cpu_device()) =
          (scratch.chip(0, 1).contract(w_m_m_s.chip(0, 0), m_m_dim) +
           scratch.chip(1, 1).contract(w_m_m_s.chip(1, 0), m_m_dim) +
           scratch.chip(2, 1).contract(w_m_m_s.chip(2, 0), m_m_dim) +
           scratch.chip(3, 1).contract(w_m_m_s.chip(3, 0), m_m_dim));
    } else {
      init_state_grad.device(ctx->eigen_cpu_device()) =
          (w_m_m.chip(0, 1).contract(scratch.chip(0, 1), m_m_dim) +
           w_m_m.chip(1, 1).contract(scratch.chip(1, 1), m_m_dim) +
           w_m_m.chip(2, 1).contract(scratch.chip(2, 1), m_m_dim) +
           w_m_m.chip(3, 1).contract(scratch.chip(3, 1), m_m_dim))
              .shuffle(array<int, 2>{1, 0});
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("VariableLSTMGrad").Device(DEVICE_CPU),
                        VariableLSTMGradOp);

REGISTER_OP("VariableLSTMGrad")
    .Input("initial_state: float32")
    .Input("initial_memory: float32")
    .Input("w_m_m: float32")
    .Input("activation: float32")
    .Input("gate_raw_act: float32")
    .Input("memory: float32")
    .Input("act_grad: float32")
    .Input("gate_raw_act_grad: float32")
    .Input("memory_grad: float32")
    .Output("input_grad: float32")
    .Output("initial_state_grad: float32")
    .Output("initial_memory_grad: float32")
    .Output("w_m_m_grad: float32")
    .Doc(R"doc(
Computes the gradient for VariableLSTM.

This is to be used conjunction with VariableLSTM. It ignores the clipping used
in the forward pass.

initial_state: 2-D with shape `[batch_size, num_nodes]`
initial_memory: 2-D with shape `[batch_size, num_nodes]`
w_m_m: 3-D with shape `[num_nodes, 4, num_nodes]`
activation: 3-D with shape `[batch_size, seq_len, num_nodes]`
gate_raw_act: 3-D with shape `[batch_size, seq_len, 4, num_nodes]`
memory: 3-D with shape `[batch_size, seq_len, num_nodes]`
act_grad: 3-D with shape `[batch_size, seq_len, num_nodes]`
gate_raw_act_grad: 3-D with shape `[batch_size, seq_len, 4, num_nodes]`
memory_grad: 3-D with shape `[batch_size, seq_len, num_nodes]`
input_grad: 3-D with shape `[batch_size, seq_len, num_nodes]`
initial_state_grad: 2-D with shape `[batch_size, num_nodes]`
initial_memory_grad: 2-D with shape `[batch_size, num_nodes]`
w_m_m_grad: 3-D with shape `[num_nodes, 4, num_nodes]`
)doc");

}  // namespace tensorflow
