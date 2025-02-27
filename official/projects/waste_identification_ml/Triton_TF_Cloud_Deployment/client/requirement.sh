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

# Update the package lists for upgrades and new package installations.
sudo apt-get update -y

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
  echo "Docker is not installed. Installing Docker..."
  # Download the Docker installation script from Docker's official site.
  curl -fsSL https://get.docker.com -o get-docker.sh

  # Run the Docker installation script.
  sudo sh get-docker.sh

  # Clean up Docker installation script
  rm -f get-docker.sh

  echo "Docker installation completed."
else
  echo "Docker is already installed. Skipping Docker installation."
fi

# Install python3-venv for creating virtual environments.
sudo apt-get install -y python3-venv python3-pip ffmpeg

# Create a virtual environment and activate it.
python3.10 -m venv myenv
source myenv/bin/activate

# Install Python packages inside the virtual environment:
pip install --no-cache-dir natsort absl-py opencv-python pandas pandas-gbq \
  google-cloud-bigquery google-auth trackpy google-cloud-storage tensorflow \
  scikit-image scikit-learn webcolors==1.13 ffmpeg-python tf_keras \
  tf_slim tritonclient[all]

# Check if the 'models' directory exists before cloning.
if [ ! -d "models" ]; then
  # Cloning project directory from TF Model Garden for postprocessing
  # and preprocessing functions.
  git clone --depth 1 https://github.com/tensorflow/models.git
else
  echo "'models' directory already exists. Skipping cloning."
fi

# Deactivate the virtual environment
deactivate

echo "Environment setup is complete."
