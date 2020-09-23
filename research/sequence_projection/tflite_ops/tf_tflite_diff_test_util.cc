/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite_ops/tf_tflite_diff_test_util.h"  // sequence_projection

#include "flatbuffers/flexbuffers.h"  // flatbuffer
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tflite {
namespace testing {

using ::tensorflow::TensorProto;
using ::testing::FloatNear;

::tflite::TensorType TfTypeToTfLiteType(::tensorflow::DataType dtype) {
  switch (dtype) {
    case ::tensorflow::DT_FLOAT:
      return TensorType_FLOAT32;

    case ::tensorflow::DT_INT32:
      return TensorType_INT32;

    case ::tensorflow::DT_STRING:
      return TensorType_STRING;

    case ::tensorflow::DT_BOOL:
      return TensorType_BOOL;

    default:
      LOG(FATAL) << "Unrecognized dtype: " << dtype;
  }
}

void SetTensorProtoShape(const std::vector<int>& shape, TensorProto* tensor) {
  auto* tensor_shape = tensor->mutable_tensor_shape();
  for (int dim : shape) {
    tensor_shape->add_dim()->set_size(dim);
  }
}

TensorProto BoolTensor(const std::vector<int>& shape,
                       const std::vector<bool>& values) {
  TensorProto tensor;
  SetTensorProtoShape(shape, &tensor);
  tensor.set_dtype(::tensorflow::DT_BOOL);
  for (bool b : values) {
    tensor.add_bool_val(b);
  }
  return tensor;
}

TensorProto IntTensor(const std::vector<int>& shape,
                      const std::vector<int>& values) {
  TensorProto tensor;
  tensor.set_dtype(::tensorflow::DT_INT32);
  SetTensorProtoShape(shape, &tensor);
  for (int i : values) {
    tensor.add_int_val(i);
  }
  return tensor;
}

TensorProto FloatTensor(const std::vector<int>& shape,
                        const std::vector<float>& values) {
  TensorProto tensor;
  tensor.set_dtype(::tensorflow::DT_FLOAT);
  SetTensorProtoShape(shape, &tensor);
  for (float f : values) {
    tensor.add_float_val(f);
  }
  return tensor;
}

TensorProto StringTensor(const std::vector<int>& shape,
                         const std::vector<std::string>& values) {
  TensorProto tensor;
  tensor.set_dtype(::tensorflow::DT_STRING);
  SetTensorProtoShape(shape, &tensor);
  for (const std::string& s : values) {
    tensor.add_string_val(s);
  }
  return tensor;
}

void TensorflowTfLiteOpTest::SetUp() {
  ConstructTensorflowOp();
  ConstructTfLiteOp();
}

void TensorflowTfLiteOpTest::ConstructTensorflowOp() {
  ::tensorflow::NodeDefBuilder builder("test_op", TensorflowOpName());
  for (const auto& attribute : GetParam().attributes) {
    builder.Attr(attribute.first, attribute.second);
  }

  int index = 0;
  for (const auto& input_tensor : GetParam().input_tensors) {
    builder.Input("input", index, input_tensor.dtype());
    index++;
  }

  TF_ASSERT_OK(builder.Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
}

void TensorflowTfLiteOpTest::RunTensorflowOp() {
  for (const auto& input_tensor : GetParam().input_tensors) {
    switch (input_tensor.dtype()) {
      case ::tensorflow::DT_FLOAT:
        AddInput<float>(input_tensor.tensor_shape(),
                        [&input_tensor](int x) -> float {
                          return input_tensor.float_val(x);
                        });
        break;

      case ::tensorflow::DT_INT32:
        AddInput<int>(
            input_tensor.tensor_shape(),
            [&input_tensor](int x) -> int { return input_tensor.int_val(x); });
        break;

      case ::tensorflow::DT_STRING:
        AddInput<::tensorflow::tstring>(
            input_tensor.tensor_shape(),
            [&input_tensor](int x) -> ::tensorflow::tstring {
              return input_tensor.string_val(x);
            });
        break;

      case ::tensorflow::DT_BOOL:
        AddInput<bool>(input_tensor.tensor_shape(),
                       [&input_tensor](int x) -> bool {
                         return input_tensor.bool_val(x);
                       });
        break;

      default:
        LOG(FATAL) << "Unrecognized dtype: " << input_tensor.DebugString();
    }
  }

  TF_ASSERT_OK(RunOpKernel());
}

std::vector<uint8_t> ConstructTfLiteCustomOptions(
    absl::flat_hash_map<std::string, ::tensorflow::AttrValue> attributes,
    const std::string& tensorflow_op) {
  // Get the default attributes of the Tensorflow op.
  const ::tensorflow::OpDef* tf_op_def;
  TF_CHECK_OK(::tensorflow::OpRegistry::Global()->LookUpOpDef(tensorflow_op,
                                                              &tf_op_def));
  for (const auto& tf_attribute : tf_op_def->attr()) {
    if (tf_attribute.has_default_value() &&
        !attributes.contains(tf_attribute.name())) {
      attributes[tf_attribute.name()] = tf_attribute.default_value();
    }
  }

  ::flexbuffers::Builder fbb;
  size_t map_start = fbb.StartMap();
  for (const auto& attribute : attributes) {
    switch (attribute.second.value_case()) {
      case ::tensorflow::AttrValue::kS:
        fbb.String(attribute.first.c_str(), attribute.second.s());
        break;

      case ::tensorflow::AttrValue::kI:
        fbb.Int(attribute.first.c_str(), attribute.second.i());
        break;

      case ::tensorflow::AttrValue::kF:
        fbb.Float(attribute.first.c_str(), attribute.second.f());
        break;

      case ::tensorflow::AttrValue::kB:
        fbb.Bool(attribute.first.c_str(), attribute.second.b());
        break;

      case ::tensorflow::AttrValue::kList: {
        int start = fbb.StartVector(attribute.first.c_str());
        if (attribute.second.list().s_size() > 0) {
          for (const std::string& s : attribute.second.list().s()) {
            fbb.String(s);
          }
        } else if (attribute.second.list().i_size() > 0) {
          for (int i : attribute.second.list().i()) {
            fbb.Int(i);
          }
        } else if (attribute.second.list().f_size() > 0) {
          for (float f : attribute.second.list().f()) {
            fbb.Float(f);
          }
        } else if (attribute.second.list().b_size() > 0) {
          for (bool b : attribute.second.list().b()) {
            fbb.Bool(b);
          }
        }
        fbb.EndVector(start, /*typed=*/true, /*fixed=*/false);
        break;
      }

      default:
        LOG(FATAL) << "Unrecognized AttrValue type: "
                   << attribute.second.DebugString();
    }
  }
  fbb.EndMap(map_start);
  fbb.Finish();

  return std::vector<uint8_t>(fbb.GetBuffer());
}

void TensorflowTfLiteOpTest::ConstructTfLiteOp() {
  std::vector<std::vector<int>> input_shapes;
  for (const auto& input_tensor : GetParam().input_tensors) {
    std::vector<int> shape;
    for (const auto& dim : input_tensor.tensor_shape().dim()) {
      shape.push_back(dim.size());
    }
    input_shapes.push_back(shape);

    tflite_inputs_.push_back(
        tflite_op_.AddInput(TfTypeToTfLiteType(input_tensor.dtype())));
  }

  for (const auto& output_tensor : GetParam().output_tensors) {
    std::vector<int> shape;
    for (const auto& dim : output_tensor.tensor.tensor_shape().dim()) {
      shape.push_back(dim.size());
    }
    if (output_tensor.quantization_params.scale != 0.0) {
      ASSERT_EQ(output_tensor.tensor.dtype(), ::tensorflow::DT_FLOAT)
          << "Quantization attempted on non-float tensor: "
          << output_tensor.tensor.DebugString();
      // We can safely use as zero min and max, as they'll be ignored and
      // the scale and zero_point will be used instead.
      tflite_outputs_.push_back(tflite_op_.AddOutput(
          {TensorType_UINT8, shape, /*min=*/0.0, /*max=*/0.0,
           output_tensor.quantization_params.scale,
           output_tensor.quantization_params.zero_point}));
    } else {
      tflite_outputs_.push_back(tflite_op_.AddOutput(
          {TfTypeToTfLiteType(output_tensor.tensor.dtype()), shape}));
    }
  }

  tflite_op_.SetCustomOp(
      TfLiteOpName(),
      ConstructTfLiteCustomOptions(GetParam().attributes, TensorflowOpName()),
      TfLiteOpRegistration());
  tflite_op_.BuildInterpreter(input_shapes);
}

void TensorflowTfLiteOpTest::RunTfLiteOp() {
  int input_index = 0;
  for (const auto& input_tensor : GetParam().input_tensors) {
    switch (input_tensor.dtype()) {
      case ::tensorflow::DT_FLOAT: {
        std::vector<float> float_val(input_tensor.float_val().begin(),
                                     input_tensor.float_val().end());
        tflite_op_.PopulateTensor<float>(tflite_inputs_[input_index],
                                         float_val);
        break;
      }

      case ::tensorflow::DT_INT32: {
        std::vector<int> int_val(input_tensor.int_val().begin(),
                                 input_tensor.int_val().end());
        tflite_op_.PopulateTensor<int>(tflite_inputs_[input_index], int_val);
        break;
      }

      case ::tensorflow::DT_STRING: {
        std::vector<std::string> string_val(input_tensor.string_val().begin(),
                                            input_tensor.string_val().end());
        tflite_op_.PopulateStringTensor(tflite_inputs_[input_index],
                                        string_val);
        break;
      }

      case ::tensorflow::DT_BOOL: {
        std::vector<bool> bool_val(input_tensor.bool_val().begin(),
                                   input_tensor.bool_val().end());
        tflite_op_.PopulateTensor<bool>(tflite_inputs_[input_index], bool_val);
        break;
      }

      default:
        LOG(FATAL) << "Unrecognized dtype: " << input_tensor.DebugString();
    }
    input_index++;
  }

  tflite_op_.Invoke();
}

void TensorflowTfLiteOpTest::CompareOpOutput() {
  for (int i = 0; i < tflite_outputs_.size(); i++) {
    const ::tensorflow::Tensor& tf_output = *GetOutput(i);
    std::vector<int> tflite_output_shape =
        tflite_op_.GetTensorShape(tflite_outputs_[i]);
    auto tf_output_shape = tf_output.shape();
    EXPECT_EQ(tf_output_shape.dims(), tflite_output_shape.size());
    for (int j = 0; j < tf_output_shape.dims(); j++) {
      EXPECT_EQ(tf_output_shape.dim_size(j), tflite_output_shape[j]);
    }

    switch (tf_output.dtype()) {
      case ::tensorflow::DT_FLOAT: {
        auto tf_output_values = tf_output.flat<float>();
        const auto& quantization_params =
            GetParam().output_tensors[i].quantization_params;
        if (quantization_params.scale != 0.0) {
          auto tflite_output_values = Dequantize(
              tflite_op_.ExtractVector<uint8_t>(tflite_outputs_[i]),
              quantization_params.scale, quantization_params.zero_point);
          for (int i = 0; i < tf_output_values.size(); i++) {
            EXPECT_THAT(
                tf_output_values(i),
                FloatNear(tflite_output_values[i], quantization_params.scale));
          }
        } else {
          auto tflite_output_values =
              tflite_op_.ExtractVector<float>(tflite_outputs_[i]);
          for (int i = 0; i < tf_output_values.size(); i++) {
            EXPECT_EQ(tf_output_values(i), tflite_output_values[i]);
          }
        }
        break;
      }

      case ::tensorflow::DT_INT32: {
        auto tf_output_values = tf_output.flat<int>();
        auto tflite_output_values =
            tflite_op_.ExtractVector<int>(tflite_outputs_[i]);
        for (int i = 0; i < tf_output_values.size(); i++) {
          EXPECT_EQ(tf_output_values(i), tflite_output_values[i]);
        }
        break;
      }

      case ::tensorflow::DT_BOOL: {
        auto tf_output_values = tf_output.flat<bool>();
        auto tflite_output_values =
            tflite_op_.ExtractVector<bool>(tflite_outputs_[i]);
        for (int i = 0; i < tf_output_values.size(); i++) {
          EXPECT_EQ(tf_output_values(i), tflite_output_values[i]);
        }
        break;
      }

      case ::tensorflow::DT_STRING: {
        auto tf_output_values = tf_output.flat<::tensorflow::tstring>();
        auto tflite_output_values =
            tflite_op_.ExtractVector<std::string>(tflite_outputs_[i]);
        for (int i = 0; i < tf_output_values.size(); i++) {
          EXPECT_EQ(tf_output_values(i), tflite_output_values[i]);
        }
        break;
      }

      default:
        LOG(FATAL) << "Unrecognized dtype: " << tf_output.dtype();
    }
  }
}

}  // namespace testing
}  // namespace tflite
