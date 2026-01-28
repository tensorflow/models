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

sudo apt-get update -y

# Install Docker if not already installed
if ! command -v docker &> /dev/null
then
  echo "Docker is not installed. Installing Docker..."
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  rm -f get-docker.sh
  echo "Docker installation completed."
else
  echo "Docker is already installed. Skipping Docker installation."
fi

# Create a virtual environment and install packages
sudo apt-get install -y python3-venv python3-pip

python3.10 -m venv myenv
source myenv/bin/activate

echo "Activated python environment, installing dependencies."

pip install --no-cache-dir natsort absl-py opencv-python pandas pandas-gbq \
  google-cloud-bigquery google-auth trackpy google-cloud-storage \
  scikit-image scikit-learn webcolors==1.13 ffmpeg-python tritonclient[all] \
  supervision==0.26.1 pillow==12.0.0

# Clone TensorFlow Model Garden if the 'models' directory does not exist
if [ ! -d "models" ]; then
  git clone --depth 1 https://github.com/tensorflow/models.git
else
  echo "'models' directory already exists. Skipping cloning."
fi

deactivate
echo "Environment setup is complete."