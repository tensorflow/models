You can deploy CircularNet using any cloud provider or recommended edge device.
However, the following instructions describe the steps to create and run a
Triton inference server deployed on Google Cloud or an NVIDIA device. The server
is designed to efficiently process an image or video as an input, splitting it
into frames and performing model predictions on every frame.

The instructions for Google Cloud are for an [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/)
computing unit, and the instructions for the edge device are for an
[NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
solution. Deploying CircularNet models requires technical expertise to manage
infrastructure, run commands, and establish connection settings and permissions
on the cloud or your edge device.