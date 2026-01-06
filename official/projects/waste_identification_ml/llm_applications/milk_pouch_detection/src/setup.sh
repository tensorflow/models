#!/bin/bash

# This script sets up the complete environment for a computer vision project
# using a model ensemble approach of bounding box detection, segmentation,
# and classification. The classification model is a pre-trained VIT model.
#
# Usage: ./setup.sh [--cuda-version cu124|cu128]
#   --cuda-version: CUDA version to use (default: cu124)

# Exit immediately if a command exits with a non-zero status.
set -o errexit
# Treat unset variables as an error when substituting.
set -o nounset
# Pipes fail if any command in the pipe fails.
set -o pipefail

# Parse command line arguments
CUDA_VERSION="cu124"
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda-version)
      CUDA_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--cuda-version cu124|cu128]"
      exit 1
      ;;
  esac
done

echo "Using CUDA version: $CUDA_VERSION"
echo "-----"

echo "🔹 Starting: Install System Dependencies"

# Remove attempts to update deprecated packages
sudo sed -i 's/^deb.*bullseye-backports/#&/' /etc/apt/sources.list
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip lsof curl
echo "✅ Finished: Install System Dependencies"
echo "-----"

echo "🔹 Starting: Create Virtual Environment"
python3.10 -m venv myenv
source myenv/bin/activate
echo "✅ Finished: Create Virtual Environment"
echo "-----"

echo "🔹 Starting: Install Torch, Torchvision, Torchaudio"
pip uninstall -y torch torchvision torchaudio > /dev/null 2>&1 || true
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
echo "✅ Finished: Install Torch, Torchvision, Torchaudio"
echo "-----"

echo "🔹 Starting: Install Grounding DINO"
git clone https://github.com/IDEA-Research/GroundingDINO.git

# Fix out of date cuda references during compile, can remove once
# https://github.com/IDEA-Research/GroundingDINO/pull/415 is merged.
cd GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
sed -i 's/value.type()/value.scalar_type()/g' ms_deform_attn_cuda.cu
sed -i 's/value.scalar_type().is_cuda()/value.is_cuda()/g' ms_deform_attn_cuda.cu

cd /home/${USER}/GroundingDINO/
pip install -e .
pip install timm==0.6.12
cd ..
echo "✅ Finished: Install Grounding DINO"
echo "-----"

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

echo "🔹 Starting: Create Project Directory Structure"
mkdir -p milk_pouch_project/models/sam2
mkdir -p milk_pouch_project/models/grounding_dino
mkdir -p milk_pouch_project/models/vit
echo "✅ Finished: Create Project Directory Structure"
echo "-----"

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

echo "🔹 Starting: Clone Required Files from TensorFlow Models Repo"
git clone --depth 1 --filter=blob:none --sparse https://github.com/tensorflow/models.git temp_tf_models
cd temp_tf_models
git sparse-checkout set official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src
cd ..
cp -r "temp_tf_models/official/projects/waste_identification_ml/llm_applications/milk_pouch_detection/src"/* milk_pouch_project/
rm -rf temp_tf_models
echo "✅ Finished: Clone Required Files from TensorFlow Models Repo"
echo "-----"

echo "🔹 Starting: Modify Imports for Local Project Structure"
find milk_pouch_project -type f -name "*.py" -exec sed -i \
  's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src.models import |from models import |g' {} +
find milk_pouch_project -type f -name "*.py" -exec sed -i \
  's|from official.projects.waste_identification_ml.llm_applications.milk_pouch_detection.src import |import |g' {} +

echo "✅ Finished: Modify Imports for Loscal Project Structure"
echo "Files downloaded and modified successfully!"
echo "-----"

echo "🎉🎉🎉 Environment setup complete! 🎉🎉🎉"
echo "-----"
