#!/bin/bash
#
# This script sets up the complete environment for a computer vision project
# using a model ensemble approach of bounding box detection, segmentation,
# and classification. The classification model is a pre-trained VIT model.

# The script performs the following steps:
# It performs the following steps:
# 1. Installs system-level dependencies.
# 2. Creates a Python virtual environment.
# 3. Installs PyTorch, GroundingDINO, SAM2, and other Python packages.
# 4. Creates project directories and downloads required model checkpoints.
# 5. Fetches the custom trained VIT classifier model.
# 6. Fetches the scripts to run the pipeline.

# Exit immediately if a command exits with a non-zero status.
set -o errexit
# Treat unset variables as an error when substituting.
set -o nounset
# Pipes fail if any command in the pipe fails.
set -o pipefail

# --- 1. Install System Dependencies ---
echo "🔹 Starting: Install System Dependencies"
apt-get update
apt-get install -y python3-venv python3-pip lsof curl
echo "✅ Finished: Install System Dependencies"
echo "-----"

# --- 2. Creating Virtual Environment ---
echo "🔹 Starting: Create Virtual Environment"
python3.10 -m venv myenv
source myenv/bin/activate
echo "✅ Finished: Create Virtual Environment"
echo "-----"

# --- 3. Install Compatible Torch version ---
echo "🔹 Starting: Install Torch, Torchvision, Torchaudio"
pip uninstall -y torch torchvision torchaudio > /dev/null 2>&1 || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ Finished: Install Torch, Torchvision, Torchaudio"
echo "-----"

# --- 4. Install Grounding DINO ---
echo "🔹 Starting: Install Grounding DINO"
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
echo "✅ Finished: Install Grounding DINO"
echo "-----"

# --- 5. Install SAM2 and Required Packages ---
echo "🔹 Starting: Install SAM2 and Required Python Packages"
pip install --no-cache-dir \
    opencv-python \
    numpy \
    ollama \
    Pillow \
    absl-py \
    natsort \
    "git+https://github.com/facebookresearch/sam2.git"
echo "✅ Finished: Install SAM2 and Required Python Packages"
echo "-----"

# --- 6. Set Up Project Directories ---
echo "🔹 Starting: Create Project Directory Structure"
mkdir -p milk_pouch_project/models/sam2
mkdir -p milk_pouch_project/models/grounding_dino
mkdir -p milk_pouch_project/models/vit
echo "✅ Finished: Create Project Directory Structure"
echo "-----"

# --- 7. Download Model Checkpoints ---
echo "🔹 Starting: Download SAM2 Checkpoint"
wget -P ./milk_pouch_project/models/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo "✅ Finished: Download SAM2 Checkpoint"
echo "-----"

echo "🔹 Starting: Download GroundingDINO Model and Config"
wget -P ./milk_pouch_project/models/grounding_dino https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P ./milk_pouch_project/models/grounding_dino https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/heads/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
echo "✅ Finished: Download GroundingDINO Model and Config"
echo "-----"

echo "🔹 Starting: Download Image Classifier Model"
wget -P ./milk_pouch_project/models/vit https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/dairy_product_packet_detection/best_vit_model_epoch_131.pt
echo "✅ Finished: Download Image Classifier Model"
echo "-----"

echo "Download the required files locally and modify the imports."
curl -sS -o milk_pouch_project/models/detection_segmentation.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/models/detection_segmentation.py
sed -i 's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import models_utils|import ../models_utils|g' detection_segmentation.py

curl -sS -o milk_pouch_project/classify_images.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/classify_images.py
sed -i 's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src.models import classification|from models import classification|g' classify_images.py

curl -sS -o milk_pouch_project/extract_objects.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/extract_objects.py
sed -i 's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src.models import detection_segmentation|from models import detection_segmentation|g' extract_objects.py
sed -i 's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import batched_io|import batched_io|g' extract_objects.py
sed -i 's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import coco_annotation_writer|import coco_annotation_writer|g' extract_objects.py

curl -sS -o milk_pouch_project/models/classification.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/models/classification.py
curl -sS -o milk_pouch_project/models/llm.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/models/llm.py
curl -sS -o milk_pouch_project/models_utils.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/models_utils.py
curl -sS -o milk_pouch_project/batched_io.py https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/batched_io.py
curl -sS -o milk_pouch_project/run_pipeline.sh https://raw.githubusercontent.com/tensorflow/models/master/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src/run_pipeline.sh
echo "Files downloaded and modified successfully!"

# --- Completion ---
echo "🎉🎉🎉 Environment setup complete! 🎉🎉🎉"
echo "-----"

# Deactivate the virtual environment
deactivate