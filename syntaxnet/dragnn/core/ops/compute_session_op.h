#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_OPS_COMPUTE_SESSION_OP_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_OPS_COMPUTE_SESSION_OP_H_

#include <string>

#include "dragnn/core/compute_session.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace syntaxnet {
namespace dragnn {

// Abstract base class: Given a MasterState and a component name, runs some op
// on the component state. The first input is always the handle. If
// OutputsHandle() is true in the derived class, then the first output will also
// be the handle.
class ComputeSessionOp : public tensorflow::OpKernel {
 public:
  explicit ComputeSessionOp(tensorflow::OpKernelConstruction *context);

  // Virtual Compute()-like function that assumes the state has been extracted
  // from the handle.
  virtual void ComputeWithState(tensorflow::OpKernelContext *context,
                                ComputeSession *compute_session) = 0;

  // Compute extracts the state from the resource manager and calls
  // ComputeWithState(). If OutputsHandle() is true, also outputs the handle for
  // subsequent ops.
  void Compute(tensorflow::OpKernelContext *context) override;

 protected:
  // If true, then the handle will be the first output of this op.
  virtual bool OutputsHandle() const = 0;

  // If true, then the constructor will check that the "component_name"
  // attribute is set.
  virtual bool RequiresComponentName() const = 0;

  // Returns the component name.
  string component_name() const {
    CHECK(RequiresComponentName());
    return component_name_;
  }

 private:
  // Name of the component used by this op.
  string component_name_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_OPS_COMPUTE_SESSION_OP_H_
