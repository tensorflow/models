This section assumes [you chose an edge device](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/#edge-device-deployment)
as the computing unit to run CircularNet models. Selecting an edge device
requires technical expertise to manage hardware and software configurations.

Selecting and configuring the right edge device significantly impacts the
performance of CircularNet ML models. To ensure optimal performance, we
recommend using NVIDIA Jetson Xavier or Jetson Orin devices, configured with a
minimum of 30 GB of RAM and between 100 GB and 200 GB of storage. This
configuration is necessary to effectively run a Triton inference server, with
which CircularNet operates because it is scalable and works with every kind of
machine learning framework.

The following are the recommended device specifications for an edge device:

<table>
  <thead>
    <tr>
      <th><strong>Device</strong></th>
      <th><strong>RAM</strong></th>
      <th><strong>Storage</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>NVIDIA Jetson series</strong>:<br>
<ul>
<li><a href="https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-series/">Jetson Xavier</a></li>
</ul>
<ul>
<li><a href="https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/">Jetson Orin</a></li>
</ul>
</td>
      <td>30 GB or more</td>
      <td>100 GB or 200 GB</td>
    </tr>
  </tbody>
</table>

To run the Triton inference server in one of the recommended Jetson devices, use
the Triton container available in the [NVIDIA DeepStream NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream-l4t).
Each device has a different JetPack version, a library that depends on the
hardware and model. The required JetPack version for this Triton container is
JetPack 6.0. When using JetPack 6.0, the container image needed is
`nvcr.io/nvidia/deepstream-l4t:7.0-triton-multiarch`.

For information about installing the latest version of JetPack on the device, see [JetPack SDK](https://developer.nvidia.com/embedded/jetpack). For other JetPack versions, refer to the [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive).

Connect [the machine vision camera](/official/projects/waste_identification_ml/circularnet-docs/content/choose-camera/) to the edge device, which leverages the graphics processing unit (GPU) to run model inference. The captured images or videos and the inference results can be streamed back to Google Cloud.

By configuring your device and completing the installation, you can ensure that
your edge device is properly configured, a crucial step in running CircularNet
models efficiently and effectively. To learn how to install CircularNet on an
edge device, see [Deploy CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/).