/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

// This file contains the implementations of the ops registered in
// grl_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// The gradient reversal op is used in domain adversarial training.  It behaves
// as the identity op during forward propagation, and multiplies its input by -1
// during backward propagation.
class GradientReversalOp : public OpKernel {
 public:
  explicit GradientReversalOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  // Gradient reversal op behaves as the identity op during forward
  // propagation. Compute() function copied from the IdentityOp::Compute()
  // function here: third_party/tensorflow/core/kernels/identity_op.h.
  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("GradientReversal").Device(DEVICE_CPU),
                        GradientReversalOp);

}  // namespace tensorflow
