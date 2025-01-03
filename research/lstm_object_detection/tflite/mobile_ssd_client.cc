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

#include "mobile_ssd_client.h"

#include <stdlib.h>

#include <map>

#include <glog/logging.h>
#include "absl/memory/memory.h"
#include "utils/conversion_utils.h"
#include "utils/ssd_utils.h"

namespace lstm_object_detection {
namespace tflite {

bool MobileSSDClient::InitializeClient(const protos::ClientOptions& options) {
  options_ = options;
  return true;
}

bool MobileSSDClient::Detect(const uint8_t* pixels, int width, int height,
                             int bytes_per_pixel, int bytes_per_row,
                             protos::DetectionResults* detections) {
  SetInputDims(width, height);
  // Grayscale input images are only compatible with grayscale models, and
  // color input images are only compatible with color models.
  CHECK((bytes_per_pixel == 1 && input_depth_ == 1) ||
        (bytes_per_pixel >= 3 && input_depth_ >= 3));
  if (HasPadding(width, height, bytes_per_pixel, bytes_per_row)) {
    std::vector<uint8_t> unpadded_pixels =
        RemovePadding(pixels, width, height, bytes_per_pixel, bytes_per_row);
    return Detect(&unpadded_pixels[0], detections);
  } else {
    return Detect(pixels, detections);
  }
}

bool MobileSSDClient::Detect(const uint8_t* pixels,
                             protos::DetectionResults* detections) {
  return BatchDetect(pixels, 1, absl::MakeSpan(&detections, 1));
}

bool MobileSSDClient::BatchDetect(
    const uint8_t* pixels, int batch_size,
    absl::Span<protos::DetectionResults*> detections) {
  if (detections.size() != batch_size) {
    LOG(ERROR) << "Batch size does not match output cardinality.";
    return false;
  }
  if (batch_size != batch_size_) {
    if (!SetBatchSize(batch_size)) {
      LOG(ERROR) << "Couldn't set batch size.";
      return false;
    }
  }
  if (!Inference(pixels)) {
    LOG(ERROR) << "Couldn't inference.";
    return false;
  }
  for (int batch = 0; batch < batch_size; ++batch) {
    if (RequiresPostProcessing()) {
      LOG(ERROR) << "Post Processing not supported.";
      return false;
    } else {
      if (!NoPostProcessNoAnchors(detections[batch])) {
        LOG(ERROR) << "NoPostProcessNoAnchors failed.";
        return false;
      }
    }
  }

  return true;
}

bool MobileSSDClient::SetBatchSize(int batch_size) {
  batch_size_ = batch_size;
  AllocateBuffers();
  if (batch_size != 1) {
    LOG(ERROR)
        << "Only single batch inference supported by default. All child "
           "classes that support batched inference should override this method "
           "and not return an error if the batch size is supported. (E.g. "
           "MobileSSDTfLiteClient).";
    return false;
  }
  return true;
}

bool MobileSSDClient::NoPostProcessNoAnchors(
    protos::DetectionResults* detections) {
  LOG(ERROR) << "not yet implemented";
  return false;
}

bool MobileSSDClient::RequiresPostProcessing() const {
  return anchors_.y_size() > 0;
}

void MobileSSDClient::SetInputDims(int width, int height) {
  CHECK_EQ(width, input_width_);
  CHECK_EQ(height, input_height_);
}

int MobileSSDClient::GetNumberOfLabels() const { return labelmap_.item_size(); }

std::string MobileSSDClient::GetLabelDisplayName(const int class_index) const {
  if (class_index < 0 || class_index >= GetNumberOfLabels()) {
    return "";
  }
  return labelmap_.item(class_index).display_name();
}

std::string MobileSSDClient::GetLabelName(const int class_index) const {
  if (class_index < 0 || class_index >= GetNumberOfLabels()) {
    return "";
  }
  return labelmap_.item(class_index).name();
}

int MobileSSDClient::GetLabelId(const int class_index) const {
  if (class_index < 0 || class_index >= GetNumberOfLabels() ||
      !labelmap_.item(class_index).has_id()) {
    return -1;
  }
  return labelmap_.item(class_index).id();
}

void MobileSSDClient::SetLabelDisplayNameInResults(
    protos::DetectionResults* detections) {
  for (auto& det : *detections->mutable_detection()) {
    for (const auto& class_index : det.class_index()) {
      det.add_display_name(GetLabelDisplayName(class_index));
    }
  }
}

void MobileSSDClient::SetLabelNameInResults(
    protos::DetectionResults* detections) {
  for (auto& det : *detections->mutable_detection()) {
    for (const auto& class_index : det.class_index()) {
      det.add_class_name(GetLabelName(class_index));
    }
  }
}

void MobileSSDClient::InitParams(const bool agnostic_mode,
                                 const bool quantize,
                                 const int num_keypoints) {
  num_keypoints_ = num_keypoints;
  code_size_ = 4 + 2 * num_keypoints;
  num_boxes_ = output_locations_size_ / code_size_;
  if (agnostic_mode) {
    num_classes_ = output_scores_size_ / num_boxes_;
  } else {
    num_classes_ = (output_scores_size_ / num_boxes_) - 1;
  }
  quantize_ = quantize;
  AllocateBuffers();
}

void MobileSSDClient::AllocateBuffers() {
  // Allocate the output vectors
  output_locations_.resize(output_locations_size_ * batch_size_);
  output_scores_.resize(output_scores_size_ * batch_size_);

  if (quantize_) {
    quantized_output_pointers_ =
        absl::make_unique<std::vector<std::unique_ptr<std::vector<uint8_t>>>>(
            batch_size_ * num_output_layers_ * 2);
    for (int batch = 0; batch < batch_size_; ++batch) {
      for (int i = 0; i < num_output_layers_; ++i) {
        quantized_output_pointers_->at(2 * (i + batch * num_output_layers_)) =
            absl::make_unique<std::vector<uint8_t>>(output_locations_sizes_[i]);
        quantized_output_pointers_->at(2 * (i + batch * num_output_layers_) +
                                       1) =
            absl::make_unique<std::vector<uint8_t>>(output_scores_sizes_[i]);
      }
    }

    quantized_output_pointers_array_.reset(
        new uint8_t*[batch_size_ * num_output_layers_ * 2]);
    for (int i = 0; i < batch_size_ * num_output_layers_ * 2; ++i) {
      quantized_output_pointers_array_[i] =
          quantized_output_pointers_->at(i)->data();
    }

    gemm_context_.set_max_num_threads(1);
  } else {
    output_pointers_[0] = output_locations_.data();
    output_pointers_[1] = output_scores_.data();
  }
}

}  // namespace tflite
}  // namespace lstm_object_detection
