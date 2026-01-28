#!/bin/bash

# Summary
cat << END
Automates the setup and deployment of a Pytorch ONNX Model on the NVIDIA Triton Inference Server.
It cleans up old setups, downloads and organizes models, creates the required configuration,
ensures screen is installed, and launches Triton in a detached session with GPU support.
END

# Check and delete the model_repository directory if it exists
if [ -d "model_repository" ]; then
  echo "Removing existing model_repository directory..."
  rm -rf model_repository
fi

# Define an associative array with model names and their URLs
declare -A models=(
  ["CircularNet_rfdetr_seg_preview_v1"]="https://storage.googleapis.com/"\
"tf_model_garden/vision/waste_identification_ml/"\
"CircularNet_rfdetr_seg_preview_v1.zip"
)

# Download, unzip, and organize models
for model_name in "${!models[@]}"; do
  url=${models[$model_name]}
  zip_file="${url##*/}"
  wget "$url" && unzip "$zip_file"
  rm "$zip_file"
done

# Install screen if not already installed
command -v screen >/dev/null 2>&1 || { \
  sudo apt update && sudo apt install -y screen; \
}

echo "Starting Triton server in a screen session."

# Start Triton server
screen -dmS server bash -c '
sudo docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v ${PWD}/model_repository:/models \
nvcr.io/nvidia/tritonserver:25.03-py3 \
tritonserver --model-repository=/models --backend-config=pytorch,version=2'