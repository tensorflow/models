+++
title = 'Choose a camera'
date = 2024-07-30T19:06:21Z
draft = true
weight = 7
+++
The quality of images captured by your camera directly impacts the accuracy of CircularNet's analysis. Therefore, selecting the right machine vision camera is crucial for successful waste identification and characterization.
  
The following are the primary key points when choosing a camera:

-  **Machine vision camera:** A specialized camera with a global shutter is recommended for optimal image quality, especially when capturing fast-moving objects on the conveyor belt. This type of camera minimizes motion blur and distortion.
-  **Image resolution:** CircularNet analyzes individual frames, so high-resolution images are essential.
-  **Installation:** Position the camera directly above the conveyor belt for consistent lighting and full coverage.

As a general rule, avoid capturing images with motion blur, fisheye effect, and quality issues due to the vibrations from the conveyor belt movement. Global shutter cameras typically meet these recommendations.

{{% panel status="primary" title="Note" icon="far fa-lightbulb" %}}The CircularNet model supports inference on individual image frames. Therefore, if the camera captures videos, the system converts those videos into frames for pre- and post-image processing.
{{% /panel %}}