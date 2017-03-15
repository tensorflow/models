#include "dragnn/core/ops/compute_session_op.h"

#include "dragnn/core/compute_session.h"
#include "dragnn/core/resource_container.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

typedef ResourceContainer<ComputeSession> ComputeSessionResource;

ComputeSessionOp::ComputeSessionOp(OpKernelConstruction *context)
    : OpKernel(context) {
  OP_REQUIRES(context, context->num_inputs() > 0,
              InvalidArgument("Must declare at least one input of type string "
                              "for the ComputeSession handle."));
  OP_REQUIRES(context, context->input_type(0) == tensorflow::DT_STRING,
              InvalidArgument("Must declare at least one input of type string "
                              "for the ComputeSession handle."));
  OP_REQUIRES_OK(context, context->GetAttr("component", &component_name_));
}

// Computes extracts the state from the resource manager and calls
// ComputeWithState(). If OutputsHandle() is true, also outputs the handle for
// subsequent ops.
void ComputeSessionOp::Compute(OpKernelContext *context) {
  // Validates the input/output tensors and the op attrs.
  if (RequiresComponentName()) {
    OP_REQUIRES(context, !component_name_.empty(),
                InvalidArgument("Required \"component\" attribute is empty."));
  }
  if (OutputsHandle()) {
    OP_REQUIRES(context, context->num_outputs() > 0,
                InvalidArgument(
                    "Must declare at least one output of type string "
                    "for the ComputeSession handle if OutputsHandle is true."));
    OP_REQUIRES(context,
                context->expected_output_dtype(0) == tensorflow::DT_STRING,
                InvalidArgument(
                    "Must declare at least one output of type string "
                    "for the ComputeSession handle if OutputsHandle is true."));
  }

  // Gets the relevant ComputeSessionResource and computes with it.
  auto handle = context->input(0).vec<string>();
  ComputeSessionResource *session_resource;
  OP_REQUIRES_OK(context,
                 context->resource_manager()->Lookup<ComputeSessionResource>(
                     handle(0), handle(1), &session_resource));
  ComputeWithState(context, session_resource->get());

  // Outputs the passed handle, if necessary, allowing op dependency chains.
  if (OutputsHandle()) {
    Tensor *output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({2}), &output));
    output->vec<string>() = handle;
  }
  session_resource->Unref();
}

}  // namespace dragnn
}  // namespace syntaxnet
