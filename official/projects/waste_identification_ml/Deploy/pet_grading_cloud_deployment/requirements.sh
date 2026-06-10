#!/bin/bash

# Summary
cat << EOF
This script sets up the environment for running ML models by ensuring Bash
execution, installing system dependencies, setting up a virtual environment,
installing ML packages, and cloning TensorFlow Model Garden.
EOF

# Ensure the script is executed with /bin/bash
if [ -z "$BASH_VERSION" ]; then
  exec /bin/bash "$0" "$@"
fi

# update linux packages
sudo apt-get update -y

# Create a virtual environment and install packages
sudo apt-get install -y python3-venv python3-pip

# Clone dinvov3 repo
git clone https://github.com/facebookresearch/dinov3.git

# Download Model weights
mkdir -p ./model_weights
[ -f ./model_weights/sam3_original_weights_sam3.pt ] || wget -P ./model_weights https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/dairy_product_packet_detection/sam3_original_weights_sam3.pt
[ -f ./model_weights/best_pet_grading_model.pth ] || wget -P ./model_weights https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/dairy_product_packet_detection/best_pet_grading_model.pth

python3.10 -m venv myenv
source myenv/bin/activate

echo "Activated python environment, installing dependencies."

pip install -r requirements.txt
pip install numpy==1.26.4

deactivate
echo "Environment setup is complete."