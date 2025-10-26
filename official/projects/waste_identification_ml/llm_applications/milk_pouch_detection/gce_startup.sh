#!/bin/bash
# This script is executed on a Google Compute Engine (GCE) instance upon startup.
# It sets up Docker, authenticates with Google Artifact Registry, pulls a
# specified Docker image, and runs it as a service.
#
# Below are the commands to debug the script and the docker container.
# 1. To debug this script, run
#   `sudo journalctl -u google-startup-scripts.service` on the VM.
# 2. To debug the docker container, run `docker ps -a` and
#   `docker logs <container_id>`.

# Exit immediately if any command fails.
set -e

echo "VM is ready. Driver is pre-installed."

echo "--- Installing Docker Engine ---"

# Install Docker using the official APT repository to ensure reliability and up-to-date packages.
# First, update package lists and install prerequisites.
apt-get update
apt-get install -y ca-certificates curl

echo "--- Setting up Environment Variables from GCE Metadata ---"
IMAGE_URI=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/IMAGE_URI -H "Metadata-Flavor: Google")
echo "IMAGE_URI: ${IMAGE_URI}"

# Create a directory for Docker's GPG key.
install -m 0755 -d /etc/apt/keyrings

# Download Docker's official GPG key.
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

# Grant read permissions for the Docker GPG key.
chmod a+r /etc/apt/keyrings/docker.asc

# Add the Docker repository to APT sources.
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update APT package lists again to include the new Docker repository.
apt-get update

# Install Docker Engine, CLI, containerd, and buildx/compose plugins.
apt-get install \
  -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Docker installed."

echo "--- Authenticating Docker with gcloud ---"

# Authenticate Docker to pull images from Google Artifact Registry.
# `gcloud` is pre-installed on Deep Learning VM images.
# This command configures Docker to use gcloud credentials for the specified registry domain.
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
echo "Docker authenticated."

# Define the Docker image and container name.
DOCKER_IMAGE="${IMAGE_URI}"
CONTAINER_NAME="milk-pouch-processor-vm"

echo "Pulling your Docker image (${DOCKER_IMAGE}) (this may take a while)..."
# Pull the specified Docker image from Google Artifact Registry.
docker pull "${DOCKER_IMAGE}"
echo "Image pull complete."

echo "--- Waiting for NVIDIA GPU driver to be ready ---"
# Loop until nvidia-smi runs successfully, indicating the driver is loaded.
# This is crucial because the startup script might run before the GPU driver
# kernel modules are fully loaded (a common race condition).
until nvidia-smi; do
  echo "Waiting for nvidia-smi to be available... (driver loading?)"
  sleep 5
done
echo "NVIDIA GPU driver is ready."

echo "Starting container loop... (This will run on the host VM)"

# This loop runs on the HOST VM, not inside the container.
# It restarts the *entire container* in each iteration.
while true; do
  echo "--- Running new container instance ---"

  # --rm ensures the container and its resources (incl. GPU)
  # are fully released upon exit.
  docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -e CATEGORY_NAME="milk_pouch" \
    -e PYTHONUNBUFFERED=1 \
    "${DOCKER_IMAGE}"|| true

  echo "Container run finished. Restarting in 10 seconds..."
  sleep 10
done