# PR Description

# Description

Implementation of the **FCOS (Fully Convolutional One-Stage Object Detection)** model as a new project in the TensorFlow Model Garden. 

This PR addresses issue #10275: **[Help wanted] FCOS: Fully Convolutional One-Stage Object Detection**.

This works corresponds to the request to implement the [FCOS](https://arxiv.org/abs/1904.01355) paper. The implementation provides an anchor-free object detection framework with the following components:
*   **Backbone**: ResNet50 (configurable via standard TF Model Garden mechanisms).
*   **FPN**: Feature Pyramid Network for multi-scale feature extraction.
*   **Heads**: Shared heads for Classification, Box Regression, and Centerness.
*   **Losses**: IOULoss, FocalCrossEntropy, and BinaryCrossEntropy for centerness.

## Type of change

- [x] A new research paper code implementation
- [x] New feature (non-breaking change which adds functionality)

## Tests

I have performed local testing to verify the integrity of the model architecture and the training loop.

*   **Unit Tests**: Added `official/projects/fcos/model/model_test.py` to verify model instantiation and forward pass shapes.
*   **Integration Test**: Ran `main.py` (training loop) locally with a subset of COCO data to verify the pipeline from data loading to gradient updates.

**Test Configuration**:
*   **Hardware**: Intel i7-1260P, NVIDIA RTX 3050 (Laptop environment)
*   **OS**: Linux
*   **TensorFlow Version**: 2.x

## Checklist

- [x] I have signed the [Contributor License Agreement](https://github.com/tensorflow/models/wiki/Contributor-License-Agreements). [User Action Required]
- [x] I have read [guidelines for pull request](https://github.com/tensorflow/models/wiki/Submitting-a-pull-request).
- [x] My code follows the [coding guidelines](https://github.com/tensorflow/models/wiki/Coding-guidelines).
- [x] I have performed a self [code review](https://github.com/tensorflow/models/wiki/Code-review) of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have made corresponding changes to the documentation (Added `README.md`).
- [x] My changes generate no new warnings.
- [x] I have added tests that prove my fix is effective or that my feature works.
