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
sudo apt-get update -qq -y
sudo apt-get install -qq -y python3-venv python3-pip lsof
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

# --- 8. Install and Configure Ollama ---
echo "🔹 Starting: Install and Start Ollama"
curl -fsSL https://ollama.com/install.sh | sh > /dev/null
ollama serve > /dev/null 2>&1 &
sleep 5
echo "✅ Finished: Install Ollama"
echo "-----"

# --- 9. Clean Up and Start Ollama ---
echo "Ensuring no old Ollama processes are running on port 11434..."
# Find any process listening on the port and stop it forcefully.
# The '2>/dev/null' suppresses errors if no process is found.
PID=$(sudo lsof -t -i:11434 2>/dev/null)
if [ -n "$PID" ]; then
  echo "Found a lingering process with PID: $PID. Stopping it..."
  sudo kill -9 $PID
  sleep 2 # Brief pause to allow the port to be released.
fi

echo "Starting the Ollama service with systemctl..."
# Use 'restart' to ensure it starts cleanly whether it was running or not.
sudo systemctl restart ollama

echo "Waiting for the Ollama API to become available..."
# This loop waits until the port is open, which is more reliable than a fixed sleep.
while ! nc -z localhost 11434; do
  sleep 1
done
echo "✅ Ollama service is responsive."


echo "🔹 Starting: Pull Ollama Gemma3 Model"
sleep 10
ollama pull gemma3:12b-it-qat > /dev/null
echo "✅ Finished: Pull Ollama Gemma3 Model"
echo "-----"

# --- 10. Verify Ollama Service ---
echo "🔹 Starting: Verify Ollama Service"
if systemctl is-active --quiet ollama; then
    echo "✅ Ollama service is active and running."
else
    echo "❌ ERROR: Ollama service failed to start. Please check status with 'systemctl status ollama'."
    exit 1
fi
echo "✅ Finished: Verify Ollama Service"
echo "-----"

# --- 11. List Pulled Ollama Models ---
echo "🔹 Starting: List Pulled Ollama Models"
ollama list
echo "✅ Finished: List Pulled Ollama Models"
echo "-----"

# --- Completion ---
echo "🎉🎉🎉 Environment setup complete! 🎉🎉🎉"
echo "-----"

# Deactivate the virtual environment
deactivate