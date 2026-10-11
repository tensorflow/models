# FCOS: Fully Convolutional One-Stage Object Detection

## Overview
This project contains an implementation of [FCOS](https://arxiv.org/abs/1904.01355) for the TensorFlow Model Garden. FCOS is an anchor-free object detection framework that proposes a per-pixel prediction fashion, analogous to semantic segmentation.

## Status: Work in Progress (Development Stage)
This repository currently hosts a **running but untrained** implementation of the FCOS model. The code is functional in terms of model architecture assembly and the training loop, but the model weights have not yet gathered meaningful patterns (i.e., it has not converged to a high mAP).

### Development & Experiments
The provided implementation is the result of focused development and experimentation conducted over the **past few months**. 

**Key Constraints & Context:**
*   **Resource Constraints:** The development environment is resource-constrained. This limits the batch sizes (currently low) and the number of training epochs we can realistically run locally.
*   **Current Focus:** The primary goal has been to establish a working pipeline (Data -> Model -> Loss -> Gradient Update) within these constraints.
*   **Next Steps:** Scaling up training on more powerful hardware (TPU/Multi-GPU) is necessary to achieve competitive object detection performance.

## Structure
*   `main.py`: Main training script.
*   `model/`: Contains the FCOS (Backbone + FPN + Head) model definitions.
*   `loss.py`: Implementation of the FCOS specific losses (IOU loss, Focal loss, Centerness loss).
*   `Data/`: Data loading and preprocessing pipelines (COCO dataset).
*   `utils/`: Utility functions.

## Usage
To run the training script from the root of the repository:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 official/projects/fcos/main.py
```

## Requirements
*   TensorFlow 2.x
*   TensorFlow Model Garden (`official`)
