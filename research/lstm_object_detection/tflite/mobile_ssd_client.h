/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_SSD_CLIENT_H_
#define TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_SSD_CLIENT_H_

#include <memory>
#include <vector>

#include <cstdint>
#include "absl/types/span.h"
#include "public/gemmlowp.h"
#include "protos/box_encodings.pb.h"
#include "protos/detections.pb.h"
#include "protos/labelmap.pb.h"
#include "protos/mobile_ssd_client_options.pb.h"

namespace lstm_object_detection {
namespace tflite {

// MobileSSDClient base class. Not thread-safe.
class MobileSSDClient {
 public:
  MobileSSDClient() = default;
  virtual ~MobileSSDClient() = default;

  // Runs detection on the image represented by 'pixels', described by the
  // associated 'width', 'height', 'bytes_per_pixel' and 'bytes_per_row'. All
  // these integers must be positive, 'bytes_per_row' must be sufficiently
  // large, and for 'bytes_per_pixel' only values 1, 3, 4 may be passed.
  // Depending on the implementation most combinations may not be allowed.
  bool Detect(const uint8_t* pixels, int width, int height, int bytes_per_pixel,
              int bytes_per_row, protos::DetectionResults* detections);

  // Same as before, but a contiguous bytewise encoding of 'pixels' is assumed.
  // That encoding can be assigned directly to the input layer of the neural
  // network.
  bool Detect(const uint8_t* pixels, protos::DetectionResults* detections);

  // Runs batched inference on the provided buffer. "pixels" is assumed to be a
  // continuous buffer of width * height * depth * batch_size pixels. It will
  // populate the detections result with batch_size DetectionResults where the
  // first result corresponds to the first image contained within the pixels
  // block. Note that not all models generalize correctly to multi-batch
  // inference and in some cases the addition of extra batches may corrupt the
  // output on the model. For example, if a network performs operations across
  // batches, BatchDetect([A, B]) may not equal [Detect(A), Detect(B)].
  bool BatchDetect(const uint8_t* pixels, int batch_size,
                   absl::Span<protos::DetectionResults*> detections);

  // Sets the dimensions of the input image on the fly, to be effective for the
  // next Detect() call.
  void SetInputDims(int width, int height);

  // Returns the width of the input image which is always positive. Usually a
  // constant or the width last set via 'SetInputDims()'.
  int GetInputWidth() const { return input_width_; }

  // Returns the height of the input image which is always positive. Usually a
  // constant or the width last set via 'SetInputDims()'.
  int GetInputHeight() const { return input_height_; }

  // Returns the depth of the input image, which is the same as bytes per pixel.
  // This will be 3 (for RGB images), 4 (for RGBA images), or 1 (for grayscale
  // images).
  int GetInputDepth() const { return input_depth_; }

  // Returns the number of possible detection labels or classes. If
  // agnostic_mode is on, then this method must return 1.
  int GetNumberOfLabels() const;

  // Returns human readable class labels given predicted class index. The range
  // of 'label_index' is determined by 'GetNumberOfLabels()'. Returns an empty
  // string if the label display name is undefined or 'label_index' is out of
  // range.
  std::string GetLabelDisplayName(const int class_index) const;

  // Returns Knowledge Graph MID class labels given predicted class index. The
  // range of 'label_index' is determined by 'GetNumberOfLabels()'. Returns an
  // empty string if the label name is undefined or 'label_index' is out of
  // range.
  std::string GetLabelName(const int class_index) const;

  // Returns the class/label ID for a given predicted class index. The range of
  // 'label_index' is determined by 'GetNumberOfLabels()'. Returns -1 in case
  // 'label_index' is out of range.
  int GetLabelId(const int class_index) const;

  // Explicitly sets human readable string class name to each detection using
  // the `display_name` field.
  void SetLabelDisplayNameInResults(protos::DetectionResults* detections);

  // Explicitly sets string class name to each detection using the `class_name`
  // fields.
  void SetLabelNameInResults(protos::DetectionResults* detections);

 protected:
  // Initializes the client from options.
  virtual bool InitializeClient(const protos::ClientOptions& options);

  // Initializes various model specific parameters.
  virtual void InitParams() {
    InitParams(false, false, 0);
  }

  virtual void InitParams(const bool agnostic_mode,
                          const bool quantize,
                          const int num_keypoints);

  virtual void InitParams(const bool agnostic_mode, const bool quantize,
                          const int num_keypoints,
                          const protos::BoxCoder& coder) {
    InitParams(agnostic_mode, quantize, num_keypoints);
    *options_.mutable_box_coder() = coder;
  }

  virtual void AllocateBuffers();

  // Sets the batch size of inference. If reimplmented, overrider is responsible
  // for calling parent (the returned status code may be ignored).
  virtual bool SetBatchSize(int batch_size);

  // Perform client specific inference on input_data.
  virtual bool Inference(const uint8_t* input_data) = 0;

  // Directly populates the results when no post-processing should take place
  // and no anchors are present. This is only possible when the TensorFlow
  // graph contains the customized post-processing ops.
  virtual bool NoPostProcessNoAnchors(protos::DetectionResults* detections);

  // Returns true iff the model returns raw output and needs its results
  // post-processed (including non-maximum suppression). If false then anchors
  // do not need to be present, LoadAnchors() can be implemented empty. Note
  // that almost all models around require post-processing.
  bool RequiresPostProcessing() const;

  // Load client specific labelmap proto file.
  virtual void LoadLabelMap() = 0;

  // Anchors for the model.
  protos::CenterSizeEncoding anchors_;
  // Labelmap for the model.
  protos::StringIntLabelMapProto labelmap_;
  // Options for the model.
  protos::ClientOptions options_;

  // Buffers for storing the model predictions
  float* output_pointers_[2];
  // The dimension of output_locations is [batch_size x num_anchors x 4]
  std::vector<float> output_locations_;
  // The dimension of output_scores is:
  //   If background class is included:
  //      [batch_size x num_anchors x (num_classes + 1)]
  //   If background class is NOT included:
  //      [batch_size x num_anchors x num_classes]
  std::vector<float> output_scores_;
  void* transient_data_;

  // Total location and score sizes.
  int output_locations_size_;
  int output_scores_size_;
  // Output location and score sizes for each output layer.
  std::vector<int> output_locations_sizes_;
  std::vector<int> output_scores_sizes_;

  // Preproccessing related parameters
  float mean_value_;
  float std_value_;
  std::vector<int> location_zero_points_;
  std::vector<float> location_scales_;
  std::vector<int> score_zero_points_;
  std::vector<float> score_scales_;

  int num_output_layers_ = 1;

  // Model related parameters
  int input_size_;
  int num_classes_;
  int num_boxes_;
  int input_width_;
  int input_height_;
  int input_depth_ = 3;  // Default value is set for backward compatibility.
  int code_size_;

  int batch_size_ = 1;  // Default value is set for backwards compatibility.

  // The number of keypoints by detection. Specific to faces for now.
  int num_keypoints_;
  // Whether to use the quantized model.
  bool quantize_;
  // The indices of restricted classes (empty if none was passed in the config).
  std::vector<int> restricted_class_indices_;

  // Buffers for storing quantized model predictions
  std::unique_ptr<std::vector<std::unique_ptr<std::vector<uint8_t>>>>
      quantized_output_pointers_;
  std::unique_ptr<uint8_t*[]> quantized_output_pointers_array_;
  gemmlowp::GemmContext gemm_context_;
};

}  // namespace tflite
}  // namespace lstm_object_detection

#endif  // TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_SSD_CLIENT_H_
