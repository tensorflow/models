# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model clients and weights for computer vision and LLM inference.

This package contains:
  - Python client classes for model inference
  - Pre-trained model weights organized by model type

Structure:
  - classification.py: ImageClassifier (ViT-B/16)
  - detection_segmentation.py: ObjectDetectionSegmentation (Grounding DINO +
  SAM2)
  - llm.py: LlmModels (using Ollama interface)

  - grounding_dino_model/: Grounding DINO weights and config
  - sam2_model/: SAM2 weights
  - image_classifier_model/: Fine-tuned ViT classifier weights
"""

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.models import classification
from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.models import detection_segmentation
from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.models import llm
