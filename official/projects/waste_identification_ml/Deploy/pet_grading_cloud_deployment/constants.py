# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Constants for the pet grading cloud deployment."""

# Specify all the input paths
DINOV3_REPO_DIR = './dinov3'
CLASSIFIER_CHECKPOINT_PATH = './model_weights/best_pet_grading_model.pth'
SAM3_CHECKPOINT_PATH = './model_weights/sam3_original_weights_sam3.pt'

# SAM3 configurations.
BATCH_SIZE = 10
DETECTION_PROMPT = 'bottles and containers'
BOTTLE_EXTRACTION_CONFIDENCE_THRESHOLD = 0.5
BOTTLE_EXTRACTION_SCORE_THRESHOLD = 0.20
BOTTLE_EXTRACTION_CONTAINMENT_THRESHOLD = 0.98
BOTTLE_EXTRACTION_MAX_SHORT_SIDE = 1024
BOTTLE_EXTRACTION_CROP_SIZE = (256, 256)


# Tracking configurations.
BYTETRACK_MINIMUM_IOU_THRESHOLD = 0.1
BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES = 2

# Classifier configuration. Must match what was used at training time.
DINOV3_MODEL_NAME = 'dinov3_vitl16'
CLASSIFICATION_THRESHOLD = 50
CLASS_NAMES = (
    'brown_bottles_grade3',
    'clean_PET_cold_drink_bottles_with_label_cap_ring_grade1',
    'clean_PET_cold_drink_bottles_without_label_cap_ring_grade1',
    'clean_PET_mango_juice_bottles_without_label_cap_ring_grade1',
    'clean_PET_water_bottles_with_label_cap_ring_grade1',
    'clean_PET_water_bottles_without_label_cap_ring_grade1',
    'clean_jars_grade1',
    'clean_liquor_bottles_without_label_cap_ring_grade1',
    'coloured_PET_bottles_grade3',
    'dirt_PET_cold_drink_bottles_with_label_cap_ring_grade3',
    'dirt_PET_cold_drink_bottles_without_label_cap_ring_grade3',
    'dirt_PET_mango_juice_bottles_without_label_cap_ring_grade3',
    'dirt_PET_water_bottles_with_label_cap_ring_grade3',
    'dirt_PET_water_bottles_without_label_cap_ring_grade3',
    'dirt_jars_grade3',
    'dirt_liquor_bottles_without_label_cap_ring_grade3',
    'full_sleeved_bottles_grade3',
    'green_bottles_grade3',
    'liquor_bottles_with_label_cap_ring_grade3',
    'non_food_bottles_grade3',
    'partially_sleeved_mango_juice_bottles_with_label_cap_ring_grade3',
)
