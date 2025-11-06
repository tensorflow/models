FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG GCS_PATH

# Set up CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9+PTX"
ENV DEBIAN_FRONTEND=noninteractive
ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Set up environment variables for the core scripts
ENV GCS_PATH=${GCS_PATH}

RUN which nvcc
RUN /usr/local/cuda/bin/nvcc --version
RUN export CUDA_HOME=/usr/local/cuda
RUN echo "CUDA_HOME: ${CUDA_HOME}"
RUN echo "GCS_PATH: ${GCS_PATH}"

# Set the user to milk_pouch to match the user in the setup.sh script.
# This is required for the correct file paths to be created.
ENV USER milk_pouch

# Install system libraries, build tools for SAM2, and gcloud CLI
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    git \
    python3-opencv \
    git-lfs \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    ca-certificates \
    ninja-build \
    cmake \
    apt-transport-https \
    gnupg \
    curl \
    sudo \
    nvidia-cuda-toolkit \
  && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
  && apt-get update -y && apt-get install -y google-cloud-cli \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /home/${USER}
ENV PATH=/usr/local/cuda/bin:$PATH

# Copy the setup.sh script into the Docker image
COPY ./src/setup.sh .
RUN chmod +x setup.sh && ./setup.sh --cuda-version cu128
RUN chmod +x milk_pouch_project/run_pipeline.sh


# Set the entrypoint to run the pipeline script in a loop.
ENTRYPOINT ["/bin/sh", "-c"]

# CMD should run the script JUST ONCE, then exit.
# The startup.sh script will handle the looping.
CMD ["./milk_pouch_project/run_pipeline.sh --gcs_path=${GCS_PATH} --batch_size=50"]
