#include <math.h>
#include <algorithm>
#include <utility>
#include <vector>

#include "dragnn/core/ops/compute_session_op.h"
#include "dragnn/core/resource_container.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

using std::vector;

using tensorflow::DEVICE_CPU;
using tensorflow::DT_FLOAT;
using tensorflow::DT_INT32;
using tensorflow::DT_INT64;
using tensorflow::DT_STRING;
using tensorflow::DataType;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::quint8;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::uint8;

namespace syntaxnet {
namespace dragnn {

namespace {

// Helper struct for resource manager.
struct VectorTriple {
  std::unique_ptr<std::vector<std::unique_ptr<std::vector<int32>>>>
      index_vectors;
  std::unique_ptr<std::vector<std::unique_ptr<std::vector<int64>>>> id_vectors;
  std::unique_ptr<std::vector<std::unique_ptr<std::vector<float>>>>
      weight_vectors;
};

}  // namespace

typedef ResourceContainer<VectorTriple> VectorTripleResource;

// See docstring in dragnn_bulk_ops.cc.
class BulkFixedFeatures : public ComputeSessionOp {
 public:
  explicit BulkFixedFeatures(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_channels", &num_channels_));

    // Input: state handle.
    vector<DataType> input_types(1, DT_STRING);

    // Output: indices, ids and weights for every fixed feature channel.
    vector<DataType> output_types;
    output_types.push_back(DT_STRING);
    for (int c = 0; c < num_channels_; ++c) output_types.push_back(DT_INT32);
    for (int c = 0; c < num_channels_; ++c) output_types.push_back(DT_INT64);
    for (int c = 0; c < num_channels_; ++c) output_types.push_back(DT_FLOAT);
    output_types.push_back(DT_INT32);
    OP_REQUIRES_OK(context, context->MatchSignature(input_types, output_types));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    constexpr int kTensorOffset = 1;
    auto indices_allocator = [context, kTensorOffset](int channel,
                                                      int num_elements) {
      Tensor *output;
      CHECK(context
                ->allocate_output(channel + kTensorOffset,
                                  TensorShape({num_elements}), &output)
                .ok());
      return output->vec<int32>().data();
    };

    const int num_channels = num_channels_;
    auto ids_allocator = [context, num_channels, kTensorOffset](
                             int channel, int num_elements) {
      Tensor *output;
      CHECK(context
                ->allocate_output(num_channels + channel + kTensorOffset,
                                  TensorShape({num_elements}), &output)
                .ok());
      return output->vec<int64>().data();
    };
    auto weights_allocator = [context, num_channels, kTensorOffset](
                                 int channel, int num_elements) {
      Tensor *output;
      CHECK(context
                ->allocate_output(2 * num_channels + channel + kTensorOffset,
                                  TensorShape({num_elements}), &output)
                .ok());
      return output->vec<float>().data();
    };

    BulkFeatureExtractor extractor(indices_allocator, ids_allocator,
                                   weights_allocator);

    int num_steps = session->BulkGetInputFeatures(component_name(), extractor);
    VLOG(2) << "Extracted " << num_steps;
    Tensor *num_steps_tensor;
    OP_REQUIRES_OK(
        context, context->allocate_output(3 * num_channels_ + 1,
                                          TensorShape({}), &num_steps_tensor));
    num_steps_tensor->scalar<int32>()() = num_steps;
  }

 private:
  // Number of fixed feature channels.
  int num_channels_;
};

REGISTER_KERNEL_BUILDER(Name("BulkFixedFeatures").Device(DEVICE_CPU),
                        BulkFixedFeatures);

// See docstring in dragnn_bulk_ops.cc.
class BulkFixedEmbeddings : public ComputeSessionOp {
 public:
  explicit BulkFixedEmbeddings(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_channels", &num_channels_));

    // Input: state handle.
    vector<DataType> input_types;
    input_types.push_back(DT_STRING);
    for (int c = 0; c < num_channels_; ++c) input_types.push_back(DT_FLOAT);
    const vector<DataType> output_types = {DT_STRING, DT_FLOAT, DT_INT32};
    OP_REQUIRES_OK(context, context->MatchSignature(input_types, output_types));
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_batch", &pad_to_batch_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_steps", &pad_to_steps_));
    use_padding_ = (pad_to_steps_ != -1) || (pad_to_batch_ != -1);
    VLOG(2) << "Created a BulkFixedEmbeddings with use_padding = "
            << use_padding_;
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    const int batch_size = session->BatchSize(component_name());
    tensorflow::ResourceMgr *rmgr = context->resource_manager();

    // Create the pool for this container, or re-use one that was allocated in a
    // previous call.
    auto create = [this](VectorTripleResource **resource) {
      LOG(INFO) << "Creating new VectorTripleResource";
      std::unique_ptr<VectorTriple> triple(new VectorTriple());
      *resource = new VectorTripleResource(std::move(triple));
      (*resource)->get()->index_vectors.reset(
          new std::vector<std::unique_ptr<std::vector<int32>>>(num_channels_));
      (*resource)->get()->id_vectors.reset(
          new std::vector<std::unique_ptr<std::vector<int64>>>(num_channels_));
      (*resource)->get()->weight_vectors.reset(
          new std::vector<std::unique_ptr<std::vector<float>>>(num_channels_));
      for (int i = 0; i < num_channels_; ++i) {
        (*resource)->get()->index_vectors->at(i).reset(
            new std::vector<int32>());
        (*resource)->get()->id_vectors->at(i).reset(new std::vector<int64>());
        (*resource)->get()->weight_vectors->at(i).reset(
            new std::vector<float>());
      }
      return Status::OK();
    };

    VectorTripleResource *vector_triple;
    auto handle = context->input(0).vec<string>();
    OP_REQUIRES_OK(context, rmgr->LookupOrCreate<VectorTripleResource>(
                                handle(0), handle(1), &vector_triple, create));

    std::vector<std::unique_ptr<std::vector<int32>>> *indices =
        vector_triple->get()->index_vectors.get();
    std::vector<std::unique_ptr<std::vector<int64>>> *ids =
        vector_triple->get()->id_vectors.get();
    std::vector<std::unique_ptr<std::vector<float>>> *weights =
        vector_triple->get()->weight_vectors.get();

    auto indices_allocator = [context, &indices](int channel, int size) {
      (*indices)[channel]->resize(size);
      return (*indices)[channel]->data();
    };
    auto ids_allocator = [context, &ids](int channel, int size) {
      (*ids)[channel]->resize(size);
      return (*ids)[channel]->data();
    };
    auto weights_allocator = [context, &weights](int channel, int size) {
      (*weights)[channel]->resize(size);
      return (*weights)[channel]->data();
    };

    BulkFeatureExtractor extractor(indices_allocator, ids_allocator,
                                   weights_allocator, use_padding_,
                                   pad_to_steps_, pad_to_batch_);

    int num_steps = session->BulkGetInputFeatures(component_name(), extractor);
    VLOG(2) << "Extracted " << num_steps;

    Tensor *num_steps_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}),
                                                     &num_steps_tensor));
    num_steps_tensor->scalar<int32>()() = num_steps;

    // Looks up and outputs embedding vectors.
    const auto &spec = session->Spec(component_name());

    int embedding_size = 0;
    for (int channel = 0; channel < num_channels_; ++channel) {
      embedding_size += context->input(1 + channel).shape().dim_size(1) *
                        spec.fixed_feature(channel).size();
    }

    const int padded_batch = std::max(pad_to_batch_, batch_size);
    const int padded_num_steps = std::max(pad_to_steps_, num_steps);
    Tensor *embedding_vectors;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            1, TensorShape({padded_num_steps * padded_batch, embedding_size}),
            &embedding_vectors));
    embedding_vectors->flat<float>().setZero();

    int channel_offset = 0;
    for (int channel = 0; channel < num_channels_; ++channel) {
      ExtractForChannel(*(indices->at(channel)), *(ids->at(channel)),
                        *(weights->at(channel)), channel_offset,
                        context->input(1 + channel), embedding_vectors);
      channel_offset += context->input(1 + channel).shape().dim_size(1) *
                        spec.fixed_feature(channel).size();
    }
    vector_triple->Unref();
  }

 private:
  void ExtractForChannel(const std::vector<int32> &indices,
                         const std::vector<int64> &ids,
                         const std::vector<float> &weights, int channel_base,
                         const Tensor &embeddings, Tensor *output) {
    // Just turn this into a feature-size matrix, then the index is just the
    // X coordinate into it. Run up the row (known length!) and sum.
    int num_elements = output->shape().dim_size(0);
    int embedding_length = embeddings.shape().dim_size(1);
    VLOG(2) << "Num elements: " << num_elements;
    VLOG(2) << "Embedding length: " << embedding_length;
    auto output_matrix = output->matrix<float>();
    auto embedding_matrix = embeddings.matrix<float>();
    VLOG(2) << "Channel base:" << channel_base;
    for (int i = 0; i < indices.size(); ++i) {
      VLOG(2) << "Feature: ind:" << indices[i] << ", id: " << ids[i]
              << ", wt: " << weights[i];
      int y_base =
          (indices[i] / num_elements) * embedding_length + channel_base;
      int x_base = indices[i] % num_elements;
      VLOG(2) << "Extracting to (x,y) = (" << x_base << "," << y_base << ")";
      for (int j = 0; j < embedding_length; ++j) {
        output_matrix(x_base, y_base + j) +=
            embedding_matrix(ids[i], j) * weights[i];
      }
    }
  }

  // Number of fixed feature channels.
  int num_channels_;

  // Will pad output to at least this many batch elements.
  int pad_to_batch_ = -1;

  // Will pad output to at least this many steps.
  int pad_to_steps_ = -1;

  // Set if either pad_to_batch or pad_to_steps is not -1.
  bool use_padding_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(BulkFixedEmbeddings);
};

REGISTER_KERNEL_BUILDER(Name("BulkFixedEmbeddings").Device(DEVICE_CPU),
                        BulkFixedEmbeddings);

// See docstring in dragnn_bulk_ops.cc.
class BulkAdvanceFromOracle : public ComputeSessionOp {
 public:
  explicit BulkAdvanceFromOracle(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_STRING}, {DT_STRING, DT_INT32}));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  // Advances all transition states along the oracle path.
  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    const int batch_size = session->BatchSize(component_name());
    const int beam_size = session->BeamSize(component_name());
    const int num_items = batch_size * beam_size;
    vector<vector<vector<int32>>> gold;

    int num_steps = 0;
    while (!session->IsTerminal(component_name())) {
      gold.emplace_back(session->EmitOracleLabels(component_name()));

      // Advance the component.
      session->AdvanceFromOracle(component_name());
      ++num_steps;
    }

    // Fills output tensor with oracle labels where possible, or -1.
    Tensor *gold_output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       1, TensorShape({num_items * num_steps}), &gold_output));
    int item = 0;
    for (int batch_ix = 0; batch_ix < batch_size; ++batch_ix) {
      for (int beam_ix = 0; beam_ix < beam_size; ++beam_ix, ++item) {
        for (int step = 0; step < num_steps; ++step) {
          gold_output->vec<int32>()(item * num_steps + step) =
              step < gold.size() ? gold[step][batch_ix][beam_ix] : -1;
        }
      }
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BulkAdvanceFromOracle);
};

REGISTER_KERNEL_BUILDER(Name("BulkAdvanceFromOracle").Device(DEVICE_CPU),
                        BulkAdvanceFromOracle);

// See docstring in dragnn_bulk_ops.cc.
template <typename T>
class BulkAdvanceFromPrediction : public ComputeSessionOp {
 public:
  explicit BulkAdvanceFromPrediction(OpKernelConstruction *context)
      : ComputeSessionOp(context) {
    const DataType dt = tensorflow::DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context,
                   context->MatchSignature({DT_STRING, dt}, {DT_STRING}));
  }

  bool OutputsHandle() const override { return true; }
  bool RequiresComponentName() const override { return true; }

  // Advances all transition states as much as possible using the given scores.
  void ComputeWithState(OpKernelContext *context,
                        ComputeSession *session) override {
    const Tensor &scores_tensor = context->input(1);
    const auto &scores = scores_tensor.matrix<T>();
    const int num_items = (session->BatchSize(component_name()) *
                           session->BeamSize(component_name()));
    const int num_actions = scores_tensor.shape().dim_size(1);
    const int num_steps = scores_tensor.shape().dim_size(0) / num_items;
    vector<float> scores_per_step(num_items * num_actions);
    for (int step = 0; step < num_steps; ++step) {
      for (int item = 0; item < num_items; ++item) {
        for (int action = 0; action < num_actions; ++action) {
          scores_per_step[item * num_actions + action] =
              scores(item * num_steps + step, action);
        }
      }
      if (!session->IsTerminal(component_name())) {
        session->AdvanceFromPrediction(component_name(), scores_per_step.data(),
                                       scores_per_step.size());
      }
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BulkAdvanceFromPrediction);
};

#define REGISTER_BULK_ADVANCE(type)                         \
  REGISTER_KERNEL_BUILDER(Name("BulkAdvanceFromPrediction") \
                              .Device(DEVICE_CPU)           \
                              .TypeConstraint<type>("T"),   \
                          BulkAdvanceFromPrediction<type>)

REGISTER_BULK_ADVANCE(float);
REGISTER_BULK_ADVANCE(quint8);
REGISTER_BULK_ADVANCE(uint8);

}  // namespace dragnn
}  // namespace syntaxnet
