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

"""Classifies images based on Image Classifier."""

import glob
import os
import shutil

from absl import app
from absl import flags
import torch
import tqdm

from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src.models import classification

FLAGS = flags.FLAGS
INPUT_DIR = "input_images"
CLASSIFICATION_DIR = "objects_for_classification"

# Path to the custom trained model for Image Classifier.
IMAGE_CLASSIFIER_WEIGHTS = "models/vit/best_vit_model_epoch_131.pt"
CLASS_NAMES = ["dairy", "other"]


def main(_) -> None:
  parent_dir = os.path.dirname(os.path.abspath(INPUT_DIR))
  predictions_dir = os.path.join(parent_dir, "predictions")

  dairy_predictions = os.path.join(predictions_dir, "dairy")
  other_predictions = os.path.join(predictions_dir, "others")
  os.makedirs(dairy_predictions, exist_ok=True)
  os.makedirs(other_predictions, exist_ok=True)

  classifier = classification.ImageClassifier(
      model_path=IMAGE_CLASSIFIER_WEIGHTS,
      class_names=CLASS_NAMES,
      device="cuda" if torch.cuda.is_available() else "cpu",
  )

  files = glob.glob(os.path.join(parent_dir, CLASSIFICATION_DIR, "*"))
  print(f"Found {len(files)} images to process...")

  total_dairy_packets = 0
  for path in tqdm.tqdm(files):
    pred_class, confidence = classifier.classify(path)
    output_filename = f"{confidence:.2f}_{os.path.basename(path)}"
    if pred_class == "dairy":
      total_dairy_packets += 1
      shutil.move(path, os.path.join(dairy_predictions, output_filename))
    else:
      shutil.move(path, os.path.join(other_predictions, output_filename))

if __name__ == "__main__":
  app.run(main)
