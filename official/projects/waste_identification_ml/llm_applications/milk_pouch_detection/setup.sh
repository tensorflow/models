#!/bin/bash
#
# This script sets up the complete environment for a computer vision project.
# It performs the following steps:
# 1. Installs system-level dependencies.
# 2. Creates a Python virtual environment.
# 3. Installs PyTorch, GroundingDINO, SAM2, and other Python packages.
# 4. Creates project directories and downloads required model checkpoints.
# 5. Installs, configures, and starts the Ollama service.
# 6. Pulls a specific LLM model for use with Ollama.

# Exit immediately if a command exits with a non-zero status.
set -o errexit
# Treat unset variables as an error when substituting.
set -o nounset
# Pipes fail if any command in the pipe fails.
set -o pipefail

# --- 1. Install System Dependencies ---
echo "🔹 Starting: Install System Dependencies"
sudo apt-get update -qq -y > /dev/null 2>&1
sudo apt-get install -qq -y python3-venv python3-pip lsof > /dev/null 2>&1
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
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ Finished: Install Torch, Torchvision, Torchaudio"
echo "-----"

# --- 4. Install Grounding DINO ---
echo "🔹 Starting: Install Grounding DINO"
git clone --quiet https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install --quiet -e .
cd ..
echo "✅ Finished: Install Grounding DINO"
echo "-----"

# --- 5. Install SAM2 and Required Packages ---
echo "🔹 Starting: Install SAM2 and Required Python Packages"
pip install --quiet --no-cache-dir \
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
mkdir -p milk_pouch_project/sam2_model
mkdir -p milk_pouch_project/grounding_dino_model
mkdir -p milk_pouch_project/image_classifier_model
echo "✅ Finished: Create Project Directory Structure"
echo "-----"

# --- 7. Download Model Checkpoints ---
echo "🔹 Starting: Download SAM2 Checkpoint"
wget -q -P ./milk_pouch_project/sam2_model https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo "✅ Finished: Download SAM2 Checkpoint"
echo "-----"

echo "🔹 Starting: Download GroundingDINO Model and Config"
wget -q -P ./milk_pouch_project/grounding_dino_model https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q -P ./milk_pouch_project/grounding_dino_model https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/heads/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
echo "✅ Finished: Download GroundingDINO Model and Config"
echo "-----"

echo "🔹 Starting: Download Image Classifier Model"
wget -q -P ./milk_pouch_project/image_classifier_model https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/dairy_product_packet_detection/best_vit_model_epoch_131.pt
echo "✅ Finished: Download Image Classifier Model"
echo "-----"

# --- Completion ---
echo "🎉🎉🎉 Environment setup complete! 🎉🎉🎉"
echo "-----"

# Deactivate the virtual environment
deactivate